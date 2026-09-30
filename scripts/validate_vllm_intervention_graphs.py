# ruff: noqa: INP001
"""Bounded eager versus CUDA-graph residual-intervention validation.

The prompt JSONL is a runtime input. The output contains only counts, digests, timings, and
aggregate metrics. Use a short private prompt set and run this through
``scripts/resource-limits.sh --gpu``. The eager engine measures the existing hook path; the graph
engine first warms up plain generation, then runs the same real, placebo, wrong-layer, and steering
legs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import logging
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoConfig, AutoTokenizer

# Keep ``python scripts/validate_vllm_intervention_graphs.py`` usable from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games.vllm_teardown import baseline_before_engine, release_engine
from reward_hacking.interp.vllm_interventions import (
    InterventionSpec,
    InterventionVLLMBackend,
    intervention,
)
from reward_hacking.model_backend import SamplingConfig
from scripts.validate_vllm_residual_interventions import (
    _vllm_prompt_logprobs,  # pyright: ignore[reportPrivateUsage]
    sample_positions,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_TOP_K = 20
DEFAULT_PROMPT_COUNT = 4
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_MAX_NUM_SEQS = 4
DEFAULT_MAX_NUM_BATCHED_TOKENS = 4096
DEFAULT_CAPTURE_SIZES = (1, 2, 4)
MAX_PROMPT_COUNT = 16
MAX_NEW_TOKENS = 256
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 16
MAX_NUM_BATCHED_TOKENS = 8192
MAX_CAPTURE_SIZE = 16
MIN_PROMPT_TOKENS = 2
RANK_TWO = 2
MIN_REPLAY_DELTA = 1e-4
POSITION_COUNT = 8
REAL_SEED = 11_037
PLACEBO_SEED = 71_003
STEERING_SEED = 130_019
STEERING_ALPHA = 20.0
WORKER_EXTENSION_CLASS = "reward_hacking.interp.vllm_interventions.ResidualInterventionWorker"
GRAPH_WORKER_CLASS = "reward_hacking.interp.vllm_graph_worker.ResidualGraphWorker"
GRAPH_ADDITIONAL_CONFIG = {"residual_intervention_graph": {"layer_ranks": {"0": 1, "1": 1}}}


@dataclass(frozen=True)
class PromptProbe:
    """Rendered prompt and token positions retained only during one validation run."""

    rendered: str
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]


@dataclass(frozen=True)
class GenerationAggregate:
    """Aggregate generation counts and post-first-token decode evidence."""

    n_outputs: int
    n_tokens: int
    min_tokens: int
    max_tokens: int
    token_digest: str
    n_decode_logprobs_after_first: int
    mean_decode_logprob_after_first: float | None
    seconds: float

    def as_dict(self) -> dict[str, object]:
        """Return JSON-safe aggregate values without per-request outputs."""
        return self.__dict__.copy()


def _load_prompts(path: Path, expected_count: int) -> tuple[str, ...]:
    """Read exactly the requested private rows; refuse implicit subsampling."""
    if not 0 < expected_count <= MAX_PROMPT_COUNT:
        raise ValueError(f"n-prompts must be in 1..{MAX_PROMPT_COUNT}")
    prompts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload: object = json.loads(line)
        if isinstance(payload, str):
            prompt = payload
        elif isinstance(payload, dict) and isinstance(payload.get("prompt"), str):
            prompt = payload["prompt"]
        else:
            raise TypeError(f"{path} line {line_number} must contain a string prompt")
        if not prompt.strip():
            raise ValueError(f"{path} line {line_number} contains an empty prompt")
        prompts.append(prompt)
    if len(prompts) != expected_count:
        raise ValueError(
            f"{path} contains {len(prompts)} rows; expected exactly {expected_count}; refusing subsampling"
        )
    return tuple(prompts)


def _prompt_digest(prompts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _make_probes(
    tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str]
) -> tuple[PromptProbe, ...]:
    probes: list[PromptProbe] = []
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template(  # pyright: ignore[reportUnknownMemberType]
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(rendered, str) or not rendered:
            raise TypeError("tokenizer chat template returned no rendered prompt")
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != RANK_TWO
            or input_ids.shape[0] != 1
        ):
            raise ValueError("tokenizer did not return one rank-2 prompt row")
        token_ids = tuple(int(token_id) for token_id in input_ids[0].tolist())
        if len(token_ids) < MIN_PROMPT_TOKENS:
            raise ValueError("each prompt must contain at least two tokens")
        positions = sample_positions(
            len(token_ids),
            hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            count=POSITION_COUNT,
        )
        probes.append(PromptProbe(rendered, token_ids, positions))
    return tuple(probes)


def _basis(d_model: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn((d_model, 1), generator=generator, dtype=torch.float32)
    return torch.linalg.qr(matrix, mode="reduced").Q.T.contiguous()


def _geometry(n_layers: int, d_model: int) -> dict[str, InterventionSpec]:
    """Make matched-rank random, wrong-layer, and large-steering specs."""
    if n_layers < RANK_TWO:
        raise ValueError("wrong-layer validation requires at least two decoder layers")
    real_basis = _basis(d_model, REAL_SEED)
    placebo_basis = _basis(d_model, PLACEBO_SEED)
    steering_vector = _basis(d_model, STEERING_SEED)[0]
    specs = {
        "real": InterventionSpec.subspace({0: real_basis}, n_layers=n_layers, d_model=d_model),
        "placebo": InterventionSpec.subspace(
            {0: placebo_basis}, n_layers=n_layers, d_model=d_model
        ),
        "wrong_layer": InterventionSpec.subspace(
            {1: real_basis}, n_layers=n_layers, d_model=d_model
        ),
        "steering": InterventionSpec.steering(
            {0: steering_vector}, alpha=STEERING_ALPHA, n_layers=n_layers, d_model=d_model
        ),
    }
    for spec in specs.values():
        spec.validate()
    return specs


def _engine_kwargs(
    *, enforce_eager: bool, max_model_len: int, max_num_seqs: int, max_num_batched_tokens: int
) -> dict[str, object]:
    """Build short bounded engine settings; graph capture sizes are deliberately tiny."""
    if not 0 < max_model_len <= MAX_MODEL_LEN:
        raise ValueError(f"max-model-len must be in 1..{MAX_MODEL_LEN}")
    if not 0 < max_num_seqs <= MAX_NUM_SEQS:
        raise ValueError(f"max-num-seqs must be in 1..{MAX_NUM_SEQS}")
    if not 0 < max_num_batched_tokens <= MAX_NUM_BATCHED_TOKENS:
        raise ValueError(f"max-num-batched-tokens must be in 1..{MAX_NUM_BATCHED_TOKENS}")
    result: dict[str, object] = {
        "enforce_eager": enforce_eager,
        "enable_prefix_caching": False,
        "worker_extension_cls": WORKER_EXTENSION_CLASS,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_logprobs": DEFAULT_TOP_K,
    }
    if not enforce_eager:
        if (
            max(DEFAULT_CAPTURE_SIZES) > max_num_seqs
            or max(DEFAULT_CAPTURE_SIZES) > MAX_CAPTURE_SIZE
        ):
            raise ValueError("CUDA-graph capture sizes exceed the bounded request size")
        result.update(
            worker_cls=GRAPH_WORKER_CLASS,
            additional_config=json.loads(json.dumps(GRAPH_ADDITIONAL_CONFIG)),
            cudagraph_capture_sizes=list(DEFAULT_CAPTURE_SIZES),
        )
    return result


def _topk_metrics(
    reference_rows: Sequence[Mapping[int, float]], comparison_rows: Sequence[Mapping[int, float]]
) -> dict[str, float | int]:
    """Compare rows while retaining no token IDs in the result."""
    if not reference_rows or len(reference_rows) != len(comparison_rows):
        raise ValueError("top-k rows must be equally sized and non-empty")
    top1 = overlap = delta = kl = 0.0
    count = shared = 0
    max_delta = 0.0
    for reference, comparison in zip(reference_rows, comparison_rows, strict=True):
        if len(reference) != DEFAULT_TOP_K or len(comparison) != DEFAULT_TOP_K:
            raise ValueError(f"top-k rows must contain exactly {DEFAULT_TOP_K} entries")
        reference_ids, comparison_ids = set(reference), set(comparison)
        overlap += len(reference_ids & comparison_ids) / DEFAULT_TOP_K
        top1 += max(reference, key=reference.__getitem__) == max(
            comparison, key=comparison.__getitem__
        )
        reference_floor, comparison_floor = min(reference.values()), min(comparison.values())
        union = reference_ids | comparison_ids
        reference_values = [reference.get(token_id, reference_floor) for token_id in union]
        comparison_values = [comparison.get(token_id, comparison_floor) for token_id in union]
        reference_max, comparison_max = max(reference_values), max(comparison_values)
        reference_norm = reference_max + math.log(
            math.fsum(math.exp(value - reference_max) for value in reference_values)
        )
        comparison_norm = comparison_max + math.log(
            math.fsum(math.exp(value - comparison_max) for value in comparison_values)
        )
        for reference_value, comparison_value in zip(
            reference_values, comparison_values, strict=True
        ):
            reference_log = reference_value - reference_norm
            kl += math.exp(reference_log) * (reference_log - comparison_value + comparison_norm)
        for token_id in reference_ids & comparison_ids:
            absolute_delta = abs(reference[token_id] - comparison[token_id])
            delta += absolute_delta
            max_delta = max(max_delta, absolute_delta)
            shared += 1
        count += 1
    if not shared:
        raise RuntimeError("top-k rows have no shared token IDs")
    return {
        "n_rows": count,
        "top1_agreement": top1 / count,
        "overlap_at_k": overlap / count,
        "mean_abs_logprob_delta": delta / shared,
        "max_abs_logprob_delta": max_delta,
        "mean_truncated_kl": max(kl / count, 0.0),
    }


def _assert_prompt_replay_effect(
    baseline: Sequence[Mapping[int, float]], intervention_rows: Sequence[Mapping[int, float]]
) -> dict[str, float | int]:
    # A strong edit can move every top-k token out of the baseline's top-k; that is the clearest
    # possible effect, but it leaves no shared token for the logprob-delta metrics.
    if len(baseline) == len(intervention_rows) and all(
        not set(reference) & set(comparison)
        for reference, comparison in zip(baseline, intervention_rows, strict=True)
    ):
        return {"n_rows": len(baseline), "top1_agreement": 0.0, "overlap_at_k": 0.0}
    metrics = _topk_metrics(baseline, intervention_rows)
    if (
        metrics["max_abs_logprob_delta"] <= MIN_REPLAY_DELTA
        and metrics["top1_agreement"] == 1.0
        and metrics["overlap_at_k"] == 1.0
    ):
        raise RuntimeError(
            "graph replay produced unchanged prompt top-k rows; intervention was skipped"
        )
    return metrics


def _digest_sequences(sequences: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for sequence in sequences:
        digest.update(json.dumps([int(token_id) for token_id in sequence]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _generate(
    backend: InterventionVLLMBackend, probes: Sequence[PromptProbe], max_new_tokens: int
) -> GenerationAggregate:
    vllm = importlib.import_module("vllm")
    sampling = vllm.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_new_tokens,
        logprobs=DEFAULT_TOP_K,
        detokenize=False,
        seed=0,
    )
    started = time.perf_counter()
    outputs = backend._llm.generate(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        [probe.rendered for probe in probes],
        sampling,
        lora_request=backend._lora_request,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    )
    elapsed = time.perf_counter() - started
    if len(outputs) != len(probes):
        raise RuntimeError("vLLM returned an unexpected number of outputs")
    sequences: list[tuple[int, ...]] = []
    decode_values: list[float] = []
    for request in outputs:
        if len(request.outputs) != 1:
            raise RuntimeError("validation expects one completion per prompt")
        completion = request.outputs[0]
        sequence = tuple(int(token_id) for token_id in completion.token_ids)
        if completion.logprobs is None or len(completion.logprobs) != len(sequence):
            raise RuntimeError("vLLM did not return one decode logprob row per token")
        for token_id, row in zip(sequence[1:], list(completion.logprobs)[1:], strict=True):
            if token_id not in row:
                raise RuntimeError("sampled token missing from vLLM decode logprobs")
            value = row[token_id].logprob
            if not math.isfinite(value):
                raise RuntimeError("vLLM returned a non-finite decode logprob")
            decode_values.append(float(value))
        sequences.append(sequence)
    lengths = [len(sequence) for sequence in sequences]
    return GenerationAggregate(
        len(sequences),
        sum(lengths),
        min(lengths),
        max(lengths),
        _digest_sequences(sequences),
        len(decode_values),
        math.fsum(decode_values) / len(decode_values) if decode_values else None,
        elapsed,
    )


def _assert_decode_replay_effect(
    baseline: GenerationAggregate, intervention_result: GenerationAggregate
) -> float | None:
    if baseline.mean_decode_logprob_after_first is None:
        raise RuntimeError("decode replay needs at least two generated tokens with logprobs")
    if intervention_result.mean_decode_logprob_after_first is None:
        # A strong edit can end every completion at its first token, leaving no decode logprobs;
        # the changed token digest is then the only, and sufficient, evidence of an effect.
        if intervention_result.token_digest == baseline.token_digest:
            raise RuntimeError(
                "decode replay was unchanged after the first token; intervention was skipped"
            )
        return None
    delta = abs(
        baseline.mean_decode_logprob_after_first
        - intervention_result.mean_decode_logprob_after_first
    )
    if delta <= MIN_REPLAY_DELTA and baseline.token_digest == intervention_result.token_digest:
        raise RuntimeError(
            "decode replay was unchanged after the first token; intervention was skipped"
        )
    return delta


def _run_leg(
    backend: InterventionVLLMBackend,
    probes: Sequence[PromptProbe],
    spec: InterventionSpec | None,
    max_new_tokens: int,
) -> tuple[tuple[dict[int, float], ...], GenerationAggregate]:
    context = nullcontext() if spec is None else intervention(backend.llm, spec)
    with context:
        rows: list[dict[int, float]] = []
        for probe in probes:
            rows.extend(
                _vllm_prompt_logprobs(
                    backend._llm,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                    probe.rendered,
                    expected_token_ids=probe.token_ids,
                    positions=probe.positions,
                    lora_request=backend._lora_request,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                )
            )
        generation = _generate(backend, probes, max_new_tokens)
    return tuple(rows), generation


def _run_engine(
    model: str,
    probes: Sequence[PromptProbe],
    kwargs: dict[str, object],
    specs: Mapping[str, InterventionSpec],
    max_new_tokens: int,
) -> dict[str, Any]:
    baseline = baseline_before_engine("vllm")
    if baseline is None:
        raise RuntimeError("vLLM engine did not provide a VRAM baseline")
    backend: InterventionVLLMBackend | None = None
    try:
        backend = InterventionVLLMBackend(
            model,
            thinking=False,
            sampling=SamplingConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_k=-1,
                seed=0,
            ),
            **kwargs,
        )
        dimensions = backend.llm.collective_rpc("residual_intervention_dimensions")
        if len(dimensions) != 1:
            raise RuntimeError(f"validation requires one worker, got {len(dimensions)}")
        n_layers, d_model = dimensions[0]
        # Time plain and edited generation on the same engine, each after its own warmup, so the
        # eager-versus-graph comparison is like for like in both conditions.
        _generate(backend, probes, max_new_tokens)
        timed_plain = _generate(backend, probes, max_new_tokens)
        with intervention(backend.llm, specs["real"]):
            _generate(backend, probes, max_new_tokens)
            timed_intervention = _generate(backend, probes, max_new_tokens)
        legs: dict[str, tuple[tuple[dict[int, float], ...], GenerationAggregate]] = {}
        for name in ("none", "real", "placebo", "wrong_layer", "steering"):
            legs[name] = _run_leg(
                backend, probes, None if name == "none" else specs[name], max_new_tokens
            )
        return {
            "n_layers": n_layers,
            "d_model": d_model,
            "timed_plain": timed_plain.as_dict(),
            "timed_intervention": timed_intervention.as_dict(),
            "legs": {name: {"rows": rows, "generation": gen} for name, (rows, gen) in legs.items()},
        }
    finally:
        if backend is not None:
            release_engine(backend, baseline_mib=baseline)
            del backend
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _compare(eager: Mapping[str, Any], graph: Mapping[str, Any]) -> dict[str, Any]:
    parity: dict[str, dict[str, float | int]] = {}
    for name in ("none", "real", "placebo", "wrong_layer", "steering"):
        parity[name] = _topk_metrics(eager["legs"][name]["rows"], graph["legs"][name]["rows"])
    graph_none = graph["legs"]["none"]
    graph_steering = graph["legs"]["steering"]
    prompt_effect = _assert_prompt_replay_effect(graph_none["rows"], graph_steering["rows"])
    decode_effect = _assert_decode_replay_effect(
        graph_none["generation"], graph_steering["generation"]
    )
    generation: dict[str, dict[str, int]] = {}
    for name in ("none", "real", "placebo", "wrong_layer", "steering"):
        eager_generation = eager["legs"][name]["generation"]
        graph_generation = graph["legs"][name]["generation"]
        generation[name] = {
            "n_outputs": min(eager_generation.n_outputs, graph_generation.n_outputs),
            "output_count_equal": int(eager_generation.n_outputs == graph_generation.n_outputs),
            "aggregate_token_digest_equal": int(
                eager_generation.token_digest == graph_generation.token_digest
            ),
        }
    return {
        "topk_parity": parity,
        "graph_replay_effect": {
            "prompt_topk": prompt_effect,
            "mean_decode_logprob_delta_after_first": decode_effect,
        },
        "generation": generation,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run eager timing, graph timing, teacher-forced parity, and decode replay checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--n-prompts", type=int, default=DEFAULT_PROMPT_COUNT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=DEFAULT_MAX_NUM_BATCHED_TOKENS
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/task16-intervention-cuda-graphs.json")
    )
    args = parser.parse_args(argv)
    if not 0 < args.max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max-new-tokens must be in 1..{MAX_NEW_TOKENS}")
    prompts = _load_prompts(args.prompts_jsonl, args.n_prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    probes = _make_probes(tokenizer, prompts)
    max_model_len = max(len(probe.token_ids) for probe in probes) + args.max_new_tokens
    config = AutoConfig.from_pretrained(args.model)
    text_config = config.text_config
    n_layers, d_model = int(text_config.num_hidden_layers), int(text_config.hidden_size)
    specs = _geometry(n_layers, d_model)
    eager_kwargs = _engine_kwargs(
        enforce_eager=True,
        max_model_len=max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    graph_kwargs = _engine_kwargs(
        enforce_eager=False,
        max_model_len=max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    if config.vision_config is not None:
        eager_kwargs["language_model_only"] = True
        graph_kwargs["language_model_only"] = True
    eager = _run_engine(args.model, probes, eager_kwargs, specs, args.max_new_tokens)
    graph = _run_engine(args.model, probes, graph_kwargs, specs, args.max_new_tokens)
    if (eager["n_layers"], eager["d_model"]) != (graph["n_layers"], graph["d_model"]):
        raise RuntimeError("eager and graph workers reported different model dimensions")
    result = {
        "schema": "vllm-intervention-cuda-graphs/v1",
        "model": args.model,
        "prompt_count": len(prompts),
        "prompt_sha256": _prompt_digest(prompts),
        "prompt_token_count": {
            "min": min(len(probe.token_ids) for probe in probes),
            "max": max(len(probe.token_ids) for probe in probes),
            "total": sum(len(probe.token_ids) for probe in probes),
        },
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "worker_extension_cls": WORKER_EXTENSION_CLASS,
            "graph_worker_cls": GRAPH_WORKER_CLASS,
            "graph_additional_config": GRAPH_ADDITIONAL_CONFIG,
            "graph_cudagraph_capture_sizes": list(DEFAULT_CAPTURE_SIZES),
            "prefix_caching": False,
        },
        "timing": {
            "eager_plain": eager["timed_plain"],
            "eager_intervention": eager["timed_intervention"],
            "graph_plain": graph["timed_plain"],
            "graph_intervention": graph["timed_intervention"],
            "warmup_included_in_timing": False,
        },
        "parity": _compare(eager, graph),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("wrote aggregate graph validation to %s", args.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    raise SystemExit(main())
