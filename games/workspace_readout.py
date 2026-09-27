"""Capture and read Jacobian-lens workspace activations for trained game arms.

The capture command is deliberately the only command that runs a transformer trunk.  It stores the
post-block residuals recorded by ``jlens.ActivationRecorder`` and a small manifest, so analyse can
be repeated on a CPU without re-running the model.  The module-level helpers are dependency-light
and are exercised by the CPU test suite.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from safetensors import safe_open

from games.framing_stimuli import framing_capture_stimuli
from games.interp_capture import render_stimuli
from games.interp_cells import sha256_of_file
from games.interp_lens_ladder import adapter_digests
from games.lora import attach_adapter, load_adapter_base, read_adapter_base_model
from games.prompts import COUNTERPART_FRAMING_IDS
from games.provenance import git_sha

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_FRAMINGS = (
    "twin",
    "another-ai",
    "human",
    "unstated",
    "human-no-shared-briefing",
    "another-ai-no-shared-briefing",
)
DEFAULT_BAND = (10, 26)
POSITIONS = ("user_end", "assistant_marker", "think_open", "early_thinking")
LENS_EVAL_PATH = str(
    Path("/var")
    / "tmp"
    / "cooperation-generalization-assets"
    / "jacobian-lens-581d398"
    / "data"
    / "evaluations"
    / "lens-eval-multihop.json"
)
MIN_READABLE_TOKEN_LENGTH = 3
MIN_MARKER_COUNT = 2
NDIM_RESIDUALS = 4
NDIM_VECTOR = 2
EARLY_THINKING_TOKENS = 20


def parse_band(value: str, *, n_layers: int) -> tuple[int, int]:
    """Parse an inclusive-looking CLI band as a half-open ``start:end`` interval."""
    match = re.fullmatch(r"(?:L)?(-?\d+)(?::|-L)(-?\d+)", value.strip())
    if match is None:
        raise ValueError(f"invalid band {value!r}; expected START:END or LSTART-LEND")
    start, end = (int(match.group(index)) for index in (1, 2))
    if start < 0:
        start += n_layers
    if end < 0:
        end += n_layers
    if not 0 <= start < end < n_layers:
        raise ValueError(f"band {value!r} resolves to [{start}, {end}), outside 0..{n_layers}")
    return start, end


def is_word_like(token: str) -> bool:
    """Whether a decoded token is suitable for the readable vocabulary tables."""
    stripped = token.strip().lstrip("Ġ▁")
    if not stripped or stripped.startswith(("##", "<")):
        return False
    if len(stripped) < MIN_READABLE_TOKEN_LENGTH or not any(
        character.isalpha() for character in stripped
    ):
        return False
    return any(character.isalpha() or "\u4e00" <= character <= "\u9fff" for character in stripped)


def bootstrap_indices(*, n_observations: int, n_resamples: int, seed: int) -> list[list[int]]:
    """Draw deterministic bootstrap index rows, sampling observations with replacement."""
    if n_observations <= 0 or n_resamples < 0:
        raise ValueError("n_observations must be positive and n_resamples non-negative")
    generator = random.Random(seed)
    return [
        [generator.randrange(n_observations) for _ in range(n_observations)]
        for _ in range(n_resamples)
    ]


def validate_manifest(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Refuse a manifest whose identity fields differ from the requested capture."""
    mismatches = {
        key: (actual.get(key), value) for key, value in expected.items() if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"manifest identity mismatch: {mismatches}")


def validate_residual(residual: torch.Tensor, *, expected_shape: Sequence[int]) -> None:
    """Validate shape, dtype-independent finiteness, and non-empty residual artifacts."""
    if tuple(residual.shape) != tuple(int(value) for value in expected_shape):
        raise ValueError(
            f"residual shape {tuple(residual.shape)} does not match {tuple(expected_shape)}"
        )
    if residual.numel() == 0 or not bool(torch.isfinite(residual.float()).all()):
        raise ValueError("residual contains no values or non-finite values")


def should_skip_cell(cell_dir: Path, identity: Mapping[str, Any]) -> bool:
    """Return true only for a complete, shape-valid cell with matching identity."""
    manifest_path = cell_dir / "manifest.json"
    residual_path = cell_dir / "residual.pt"
    if not manifest_path.is_file() or not residual_path.is_file():
        return False
    manifest = cast("dict[str, Any]", json.loads(manifest_path.read_text(encoding="utf-8")))
    try:
        validate_manifest(manifest, identity)
        residual = torch.load(residual_path, map_location="cpu", weights_only=True)
        validate_residual(residual, expected_shape=manifest["residual_shape"])
    except (OSError, TypeError, ValueError, RuntimeError, KeyError):
        return False
    return True


def single_token_concept_ids(
    tokenizer: PreTrainedTokenizerBase,
    concept_sets: Mapping[str, Sequence[str]],
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[str, ...]]]:
    """Map words to unique token ids, trying spacing and capitalization variants.

    The second return value records words that had no single-token variant.  It is written to the
    analysis report rather than silently treating a multi-token word as one of its fragments.
    """
    ids_by_set: dict[str, tuple[int, ...]] = {}
    skipped: dict[str, tuple[str, ...]] = {}
    for set_name, words in concept_sets.items():
        found: set[int] = set()
        omitted: list[str] = []
        for word in words:
            variants = (word, f" {word}", word.capitalize(), f" {word.capitalize()}")
            candidates: set[int] = set()
            for variant in variants:
                encoded = tokenizer(variant, add_special_tokens=False)["input_ids"]
                token_ids = [int(value) for value in encoded]
                if len(token_ids) == 1:
                    candidates.add(token_ids[0])
            if candidates:
                found.update(candidates)
            else:
                omitted.append(word)
        ids_by_set[set_name] = tuple(sorted(found))
        skipped[set_name] = tuple(omitted)
    return ids_by_set, skipped


def _token_ids(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [int(value) for value in encoded]


def find_prompt_positions(
    tokenizer: PreTrainedTokenizerBase,
    rendered: str,
    token_ids: Sequence[int] | None = None,
) -> dict[str, int]:
    """Locate prompt boundaries from the actual rendered token ids, never hardcoded offsets."""
    ids = list(token_ids) if token_ids is not None else _token_ids(tokenizer, rendered)
    marker_ids: dict[str, int] = {}
    for name, token in (
        ("im_start", "<|im_start|>"),
        ("im_end", "<|im_end|>"),
        ("think", "<think>"),
        ("newline", "\n"),
    ):
        if token == "\n":
            newline_ids = _token_ids(tokenizer, token)
            if len(newline_ids) != 1:
                raise ValueError("newline marker is not one token for this tokenizer")
            marker_ids[name] = newline_ids[0]
        else:
            converted = tokenizer.convert_tokens_to_ids(token)
            if isinstance(converted, list):
                raise TypeError(f"marker {token!r} does not map to one token id")
            marker_ids[name] = int(converted)
    assistant_starts = [
        index for index, token_id in enumerate(ids) if token_id == marker_ids["im_start"]
    ]
    if len(assistant_starts) < MIN_MARKER_COUNT:
        raise ValueError("rendered prompt has no distinct user and assistant turn markers")
    assistant_start = assistant_starts[-1]
    if assistant_start == 0 or ids[assistant_start - 1] != marker_ids["newline"]:
        raise ValueError("assistant turn marker is not preceded by the user turn terminator")
    user_end_markers = [
        index
        for index, token_id in enumerate(ids[:assistant_start])
        if token_id == marker_ids["im_end"]
    ]
    think_markers = [
        index
        for index, token_id in enumerate(ids[assistant_start:], start=assistant_start)
        if token_id == marker_ids["think"]
    ]
    if len(user_end_markers) != 1 or len(think_markers) != 1:
        raise ValueError("expected exactly one user end marker and one final think opener")
    user_end_marker = user_end_markers[0]
    think_marker = think_markers[0]
    if user_end_marker == 0 or think_marker + 1 >= len(ids):
        raise ValueError("rendered prompt has no content token or think-opening newline")
    return {
        "user_end": user_end_marker - 1,
        "assistant_marker": assistant_start + 1,
        "think_open": think_marker + 1,
    }


@dataclass(frozen=True)
class LoadedLens:
    """CPU representation of a fitted Jacobian lens."""

    jacobians: dict[int, torch.Tensor]
    n_prompts: int
    d_model: int

    @property
    def source_layers(self) -> tuple[int, ...]:
        """Return fitted source layers in ascending order."""
        return tuple(sorted(self.jacobians))

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Transport residual vectors at one source layer into final-layer space."""
        if layer not in self.jacobians:
            raise KeyError(f"layer {layer} is not in the fitted lens")
        return residual.float() @ self.jacobians[layer].to(residual.device).float().T


def load_lens(path: Path) -> LoadedLens:
    """Load a jlens ``lens.pt`` without importing the optional jlens package."""
    payload = cast("dict[str, Any]", torch.load(path, map_location="cpu", weights_only=True))
    if {"J", "n_prompts", "d_model"} - set(payload):
        raise ValueError(f"{path} is missing the jlens J/n_prompts/d_model fields")
    jacobians = {int(layer): tensor.float() for layer, tensor in payload["J"].items()}
    d_model = int(payload["d_model"])
    if not jacobians or any(
        tuple(matrix.shape) != (d_model, d_model) for matrix in jacobians.values()
    ):
        raise ValueError(f"{path} has inconsistent Jacobian matrix shapes")
    return LoadedLens(jacobians=jacobians, n_prompts=int(payload["n_prompts"]), d_model=d_model)


@dataclass(frozen=True)
class StoredUnembed:
    """The Qwen final RMSNorm and lm-head, loaded without the transformer trunk."""

    norm_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    rms_eps: float

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Apply the model's final norm and vocabulary projection."""
        input_dtype = self.lm_head_weight.dtype
        values = residual.to(device=self.lm_head_weight.device, dtype=input_dtype).float()
        normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.rms_eps)
        normalized = normalized * (1.0 + self.norm_weight.float())
        return normalized.to(input_dtype).float() @ self.lm_head_weight.float().T


def load_unembed(model_path: Path) -> StoredUnembed:
    """Load only final norm and lm-head tensors from a local safetensors checkpoint."""
    config = cast(
        "dict[str, Any]", json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    )
    text_config = cast("dict[str, Any]", config.get("text_config", config))
    epsilon = float(text_config.get("rms_norm_eps", 1e-6))
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        logger.warning(
            "no safetensors index at %s; loading the full checkpoint on CPU to extract the unembed",
            index_path,
        )
        transformers = importlib.import_module("transformers")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cpu"
        )
        decoder = (
            model.model.language_model if hasattr(model.model, "language_model") else model.model
        )
        norm = decoder.norm.weight.detach().cpu().contiguous()
        head = model.lm_head.weight.detach().cpu().contiguous()
        del model
        return StoredUnembed(norm_weight=norm, lm_head_weight=head, rms_eps=epsilon)
    index = cast("dict[str, Any]", json.loads(index_path.read_text(encoding="utf-8")))
    weight_map = cast("dict[str, str]", index["weight_map"])
    norm_candidates = [
        key
        for key in (
            "model.language_model.norm.weight",
            "model.norm.weight",
            "language_model.norm.weight",
        )
        if key in weight_map
    ]
    head_candidates = [
        key for key in ("lm_head.weight", "model.lm_head.weight") if key in weight_map
    ]
    if len(norm_candidates) != 1 or len(head_candidates) != 1:
        raise ValueError(
            f"expected one final norm and lm head, found {norm_candidates} and {head_candidates}"
        )
    norm_key = norm_candidates[0]
    head_key = head_candidates[0]
    norm_path = model_path / weight_map[norm_key]
    head_path = model_path / weight_map[head_key]
    with safe_open(str(norm_path), framework="pt", device="cpu") as handle:
        norm = handle.get_tensor(norm_key).contiguous()
    with safe_open(str(head_path), framework="pt", device="cpu") as handle:
        head = handle.get_tensor(head_key).contiguous()
    if norm.ndim != 1 or head.ndim != NDIM_VECTOR or norm.shape[0] != head.shape[1]:
        raise ValueError(
            f"unembed tensors have incompatible shapes {tuple(norm.shape)} and {tuple(head.shape)}"
        )
    return StoredUnembed(norm_weight=norm, lm_head_weight=head, rms_eps=epsilon)


def _load_jlens() -> ModuleType:
    try:
        return importlib.import_module("jlens")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "jlens is unavailable; set PYTHONPATH to the pinned jacobian-lens clone"
        ) from exc


@dataclass(frozen=True)
class CaptureContext:
    """Inputs shared by one arm's GPU capture pass."""

    generation_model: Any
    tokenizer: PreTrainedTokenizerBase
    jl_model: Any
    lens: LoadedLens
    rendered: Mapping[str, str]
    stimuli: Sequence[Any]
    source_layers: Sequence[int]
    device: torch.device
    arm: str
    out_dir: Path
    reference_lens: Any


def _capture_arm(context: CaptureContext) -> None:
    activation_recorder = importlib.import_module("jlens.hooks").ActivationRecorder

    rows: list[torch.Tensor] = []
    manifest_rows: list[dict[str, Any]] = []
    for stimulus in context.stimuli:
        prompt = context.rendered[stimulus.stimulus_id]
        encoded_ids = context.jl_model.encode(prompt, max_length=4096)
        ids = encoded_ids.to(context.device)
        positions = find_prompt_positions(context.tokenizer, prompt, encoded_ids[0].tolist())
        with torch.inference_mode():
            generated = context.generation_model.generate(
                input_ids=ids,
                do_sample=False,
                max_new_tokens=EARLY_THINKING_TOKENS,
                pad_token_id=context.tokenizer.eos_token_id,
            )
        generated_ids = generated[0, ids.shape[1] :]
        all_ids = generated
        record_at = list(context.source_layers)
        with activation_recorder(context.jl_model.layers, at=record_at) as recorder:
            context.jl_model.forward(all_ids)
        captured = {
            layer: recorder.activations[layer][0].detach() for layer in context.source_layers
        }
        early_start = ids.shape[1]
        early_end = early_start + int(generated_ids.shape[0])
        if generated_ids.shape[0] != EARLY_THINKING_TOKENS:
            raise ValueError(
                f"arm {context.arm} generated {generated_ids.shape[0]} tokens instead of "
                f"{EARLY_THINKING_TOKENS} "
                f"for {stimulus.stimulus_id}"
            )
        vectors = []
        for layer in context.source_layers:
            activation = captured[layer]
            selected = torch.stack(
                (
                    activation[positions["user_end"]],
                    activation[positions["assistant_marker"]],
                    activation[positions["think_open"]],
                    activation[early_start:early_end].float().mean(dim=0).to(activation.dtype),
                )
            )
            vectors.append(selected)
        row = torch.stack(vectors, dim=1)
        validate_residual(
            row, expected_shape=(len(POSITIONS), len(context.source_layers), context.lens.d_model)
        )
        rows.append(row.to(dtype=torch.bfloat16, device="cpu"))
        if len(rows) == 1 and context.arm == "base":
            # Mid-depth, where the lens is meant to be read; layer 0 would pass on near-noise.
            check_layer = context.source_layers[len(context.source_layers) // 2]
            reference_logits, _, _ = context.reference_lens.apply(
                context.jl_model,
                prompt,
                layers=[check_layer],
                positions=[positions["think_open"]],
                max_seq_len=4096,
                use_jacobian=True,
            )
            captured_logits = (
                context.jl_model.unembed(
                    context.reference_lens.transport(
                        # fp32, exactly as lens.apply's own select() casts before transporting
                        captured[check_layer][positions["think_open"]].float(),
                        check_layer,
                    )
                )
                .float()
                .cpu()
            )
            if not torch.allclose(
                captured_logits,
                reference_logits[check_layer][0],
                rtol=3e-2,
                atol=3e-2,
            ):
                raise ValueError(
                    "jlens capture residual does not match lens.apply at the think_open position"
                )
        manifest_rows.append(
            {
                "stimulus_id": stimulus.stimulus_id,
                "side": stimulus.side,
                "pair_id": stimulus.pair_id,
                "arm": context.arm,
                "position_names": list(POSITIONS),
                "token_indices": {
                    **positions,
                    "early_thinking": list(range(early_start, early_end)),
                },
                "generated_early_thinking": context.tokenizer.decode(generated_ids.tolist()),
            }
        )
    residuals = torch.stack(rows)
    arm_dir = context.out_dir / context.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.save(residuals, arm_dir / "residuals.pt")
    (arm_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )


def accepted_adapter_base(
    arm: str, adapter_dir: Path, base_model: str, aliases: Sequence[str]
) -> str:
    """Return the base id an adapter records, refusing it unless it names the loaded base or an alias."""
    recorded_base = read_adapter_base_model(adapter_dir).rstrip("/")
    accepted_bases = {str(base_model).rstrip("/"), *aliases}
    if recorded_base not in accepted_bases:
        raise ValueError(
            f"arm {arm}: adapter records base {recorded_base!r}, which is neither the loaded "
            f"base {base_model!r} nor a declared --base-model-alias {sorted(aliases)}"
        )
    return recorded_base


def capture(args: argparse.Namespace) -> None:
    """Run the GPU capture pass and write one residual artifact per arm."""
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("capture requires --device cuda; this command must be run on the GPU")
    framings = tuple(args.framings)
    unknown = sorted(set(framings) - set(COUNTERPART_FRAMING_IDS))
    if unknown:
        raise ValueError(f"unknown framing ids {unknown}; expected {list(COUNTERPART_FRAMING_IDS)}")
    if not args.lens.is_file():
        raise FileNotFoundError(args.lens)
    lens = load_lens(args.lens)
    jlens = _load_jlens()
    tokenizer = __import__(
        "transformers", fromlist=["AutoTokenizer"]
    ).AutoTokenizer.from_pretrained(args.base_model)
    stimuli = framing_capture_stimuli(framings)
    rendered = render_stimuli(
        tokenizer,
        stimuli,
        convention="templated_here",
        enable_thinking=True,
        include_assistant_prefix=False,
    )
    model = load_adapter_base(args.base_model, dtype=torch.bfloat16, device=torch.device("cuda"))
    jl_model = jlens.from_hf(model, tokenizer)
    reference_lens = jlens.JacobianLens.load(str(args.lens))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    arms: dict[str, Path | None] = {"base": None}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        if not name or not path:
            raise ValueError(f"invalid --arm {spec!r}; expected name=adapter_dir")
        if name in arms:
            raise ValueError(f"duplicate arm name {name!r}")
        arms[name] = Path(path)
    attached = None
    for arm, adapter_dir in arms.items():
        if adapter_dir is not None:
            recorded_base = accepted_adapter_base(
                arm, adapter_dir, args.base_model, args.base_model_alias
            )
            attached = attach_adapter(
                model,
                adapter_dir,
                recorded_base,
                existing=attached.peft_model if attached else None,
            )
            generation_model = attached.peft_model
        else:
            generation_model = model
        arm_dir = out_dir / arm
        if _arm_artifact_complete(arm_dir, lens=lens, n_stimuli=len(stimuli)):
            logger.info("resume: skipping complete arm %s", arm)
            continue
        _capture_arm(
            CaptureContext(
                generation_model=generation_model,
                tokenizer=tokenizer,
                jl_model=jl_model,
                lens=lens,
                rendered=rendered,
                stimuli=stimuli,
                source_layers=lens.source_layers,
                device=torch.device("cuda"),
                arm=arm,
                out_dir=out_dir,
                reference_lens=reference_lens,
            )
        )
    adapter_provenance = {
        name: (
            None
            if path is None
            else {
                "adapter_weights_sha256": adapter_digests(path)[0],
                "adapter_config_sha256": adapter_digests(path)[1],
            }
        )
        for name, path in arms.items()
    }
    provenance = {
        "git_sha": git_sha(),
        "lens_path": str(args.lens),
        "lens_sha256": sha256_of_file(args.lens),
        "base_model": args.base_model,
        "base_model_aliases": sorted(args.base_model_alias),
        "jlens_commit": "581d398",
        "source_layers": list(lens.source_layers),
        "target_layer": args.target_layer,
        "framings": list(framings),
        "arms": {name: (str(path) if path is not None else None) for name, path in arms.items()},
        "adapter_digests": adapter_provenance,
    }
    (out_dir / "run.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _arm_artifact_complete(arm_dir: Path, *, lens: LoadedLens, n_stimuli: int) -> bool:
    """Check that a resumable arm has both a valid tensor and one manifest row per stimulus."""
    residual_path = arm_dir / "residuals.pt"
    manifest_path = arm_dir / "manifest.jsonl"
    if not residual_path.is_file() or not manifest_path.is_file():
        return False
    try:
        residuals = torch.load(residual_path, map_location="cpu", weights_only=True)
        manifest = _read_manifest(manifest_path)
        validate_residual(
            residuals,
            expected_shape=(n_stimuli, len(POSITIONS), len(lens.source_layers), lens.d_model),
        )
    except (OSError, TypeError, ValueError, RuntimeError, KeyError):
        return False
    return len(manifest) == n_stimuli


def decode_vocab(tokenizer: PreTrainedTokenizerBase, n_rows: int) -> list[str]:
    """Label every lm_head row, special tokens and padding rows included.

    ``vocab_size`` omits the added special tokens, and the lm_head carries padding rows the tokenizer
    never emits; both get labels so no top-k index can fall off the end.
    """
    n_tokens = len(tokenizer)
    if n_tokens > n_rows:
        raise ValueError(f"tokenizer has {n_tokens} ids but the lm_head only {n_rows} rows")
    decoded = [str(tokenizer.decode([index])) for index in range(n_tokens)]
    return decoded + [f"<lm_head padding row {index}>" for index in range(n_tokens, n_rows)]


def _bootstrap_ci(
    values: Sequence[float], *, seed: int, n_resamples: int = 1000
) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    indices = bootstrap_indices(n_observations=len(values), n_resamples=n_resamples, seed=seed)
    means = sorted(sum(values[index] for index in sample) / len(sample) for sample in indices)
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _rank_percentile(logits: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """Return top-is-one rank percentiles for selected vocabulary ids."""
    vocabulary_size = logits.shape[-1]
    ranks = logits.argsort(dim=-1, descending=True).argsort(dim=-1)
    denominator = max(1, vocabulary_size - 1)
    return 1.0 - ranks[:, token_ids].float() / denominator


def rank_token_shifts(
    mean_delta: torch.Tensor,
    fraction_positive: torch.Tensor,
    vocab: Sequence[str],
    *,
    sign: float,
    top_k: int = 40,
) -> list[dict[str, Any]]:
    """Top tokens by signed mean shift; NaN entries (filtered vocabulary) never rank."""
    signed = (sign * mean_delta).nan_to_num(nan=float("-inf"))
    values, token_indices = signed.topk(top_k)
    return [
        {
            "token": vocab[int(token_index)],
            "delta": float(sign * value),
            "fraction_positive": float(fraction_positive[int(token_index)]),
        }
        for value, token_index in zip(values, token_indices, strict=True)
        if value > 0
    ]


def analyse(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915
    """Read captured residuals and write aggregate token and concept tables."""
    lens = load_lens(args.lens)
    run = cast(
        "dict[str, Any]", json.loads((args.capture_dir / "run.json").read_text(encoding="utf-8"))
    )
    declared_target = int(run.get("target_layer", max(lens.source_layers) + 1))
    band_start, band_end = parse_band(
        args.band, n_layers=max(declared_target + 1, max(lens.source_layers) + 1)
    )
    layers = tuple(layer for layer in lens.source_layers if band_start <= layer < band_end)
    if not layers:
        raise ValueError(f"band {args.band!r} contains no fitted source layers")
    band_positions = [lens.source_layers.index(layer) for layer in layers]
    tokenizer = __import__(
        "transformers", fromlist=["AutoTokenizer"]
    ).AutoTokenizer.from_pretrained(args.model_path)
    unembed = load_unembed(args.model_path)
    if args.device != "cpu":
        target_device = torch.device(args.device)
        unembed = StoredUnembed(
            norm_weight=unembed.norm_weight.to(target_device),
            lm_head_weight=unembed.lm_head_weight.to(target_device),
            rms_eps=unembed.rms_eps,
        )
    arm_names = sorted(run["arms"])
    captures: dict[str, tuple[torch.Tensor, list[dict[str, Any]]]] = {}
    for arm in arm_names:
        residual_path = args.capture_dir / arm / "residuals.pt"
        manifest_path = args.capture_dir / arm / "manifest.jsonl"
        residuals = torch.load(residual_path, map_location="cpu", weights_only=True)
        manifests = _read_manifest(manifest_path)
        if residuals.ndim != NDIM_RESIDUALS or residuals.shape[0] != len(manifests):
            raise ValueError(f"arm {arm} residual/manifest lengths disagree")
        captures[arm] = (residuals.float(), manifests)
    base_residuals, base_manifest = captures["base"]
    vocab = decode_vocab(tokenizer, int(unembed.lm_head_weight.shape[0]))
    readable = [is_word_like(token) for token in vocab]
    mean_logits: dict[str, torch.Tensor] = {}
    report: dict[str, Any] = {
        "band": [band_start, band_end],
        "top_tokens": {},
        "paired_differences": {},
        "concept_sets": {},
    }
    concepts: dict[str, list[str]] = {}
    if args.concept_sets is not None:
        concepts = cast(
            "dict[str, list[str]]", json.loads(args.concept_sets.read_text(encoding="utf-8"))
        )
    concept_ids, skipped = single_token_concept_ids(tokenizer, concepts) if concepts else ({}, {})
    if skipped:
        logger.info("concept words skipped because no single-token variant exists: %s", skipped)
    report["concept_skipped_words"] = skipped
    sides = sorted({str(row["side"]) for row in base_manifest})
    for arm, (residuals, manifests) in captures.items():
        for side in sides:
            indices = [index for index, row in enumerate(manifests) if row["side"] == side]
            if not indices:
                continue
            for position_index, position_name in enumerate(manifests[0]["position_names"]):
                layer_means: list[torch.Tensor] = []
                band_per_stimulus: list[torch.Tensor] = []
                for layer in lens.source_layers:
                    logits = unembed.unembed(
                        lens.transport(
                            residuals[indices, position_index, lens.source_layers.index(layer)],
                            layer,
                        )
                    )
                    layer_means.append(logits.mean(dim=0))
                    if layer in layers:
                        band_per_stimulus.append(logits)
                means = torch.stack(layer_means)
                mean_logits[f"{arm}|{side}|{position_name}"] = means.to(dtype=torch.float16)
                top_values, top_indices = (
                    means[band_positions].mean(dim=0).topk(min(30, means.shape[-1]))
                )
                report["top_tokens"][f"{arm}|{side}|{position_name}"] = [
                    {"token": vocab[int(index)], "logit": float(value)}
                    for value, index in zip(top_values, top_indices, strict=True)
                ]
                if arm == "base" and concept_ids:
                    base_concept_report: dict[str, Any] = {}
                    for concept_name, ids in concept_ids.items():
                        if not ids:
                            continue
                        base_profile: list[dict[str, float]] = []
                        rank_values: list[float] = []
                        logit_values: list[float] = []
                        for layer_logits, layer in zip(band_per_stimulus, layers, strict=True):
                            rank_value = float(_rank_percentile(layer_logits, ids).mean())
                            logit_value = float(layer_logits[:, ids].mean())
                            rank_values.append(rank_value)
                            logit_values.append(logit_value)
                            base_profile.append(
                                {
                                    "layer": float(layer),
                                    "base_rank_percentile": rank_value,
                                    "base_mean_logit": logit_value,
                                }
                            )
                        base_concept_report[concept_name] = {
                            "base_band_mean": sum(rank_values) / len(rank_values),
                            "base_mean_logit": sum(logit_values) / len(logit_values),
                            "profile": base_profile,
                        }
                    report["concept_sets"][f"{arm}|{side}|{position_name}"] = base_concept_report
                if arm != "base":
                    base_indices = [
                        index for index, row in enumerate(base_manifest) if row["side"] == side
                    ]
                    if len(base_indices) != len(indices):
                        raise ValueError(f"arm {arm} is not paired with base for framing {side}")
                    if [base_manifest[index]["stimulus_id"] for index in base_indices] != [
                        manifests[index]["stimulus_id"] for index in indices
                    ]:
                        raise ValueError(
                            f"arm {arm} stimulus order does not match base for framing {side}"
                        )
                    # Log-probabilities, so a change in overall logit scale or offset between
                    # arms cannot masquerade as a change in which tokens the workspace promotes.
                    trained_band = torch.stack(
                        [layer_logits.log_softmax(dim=-1) for layer_logits in band_per_stimulus]
                    ).mean(dim=0)
                    base_band = torch.stack(
                        [
                            unembed.unembed(
                                lens.transport(
                                    base_residuals[
                                        base_indices,
                                        position_index,
                                        lens.source_layers.index(layer),
                                    ],
                                    layer,
                                )
                            ).log_softmax(dim=-1)
                            for layer in layers
                        ]
                    ).mean(dim=0)
                    delta = trained_band - base_band
                    readable_mask = torch.tensor(readable, device=delta.device)
                    mean_delta = delta.mean(dim=0).masked_fill(~readable_mask, float("nan"))
                    fraction_positive = (delta > 0).float().mean(dim=0)
                    risen = rank_token_shifts(mean_delta, fraction_positive, vocab, sign=1.0)
                    fallen = rank_token_shifts(mean_delta, fraction_positive, vocab, sign=-1.0)
                    report["paired_differences"][f"{arm}|{side}|{position_name}"] = {
                        "risen": risen,
                        "fallen": fallen,
                        "filter": "decoded length >=3 and contains a letter or CJK character",
                        "scale": "band-mean log-softmax, trained minus base, paired by stimulus",
                    }
                    if concept_ids:
                        concept_report: dict[str, Any] = {}
                        for concept_name, ids in concept_ids.items():
                            if not ids:
                                continue
                            trained_scores: list[float] = []
                            base_scores: list[float] = []
                            trained_logits_means: list[float] = []
                            base_logits_means: list[float] = []
                            profile: list[dict[str, float]] = []
                            pair_values: dict[str, list[float]] = {}
                            for layer in layers:
                                trained_logits = unembed.unembed(
                                    lens.transport(
                                        residuals[
                                            indices, position_index, lens.source_layers.index(layer)
                                        ],
                                        layer,
                                    )
                                )
                                base_logits = unembed.unembed(
                                    lens.transport(
                                        base_residuals[
                                            base_indices,
                                            position_index,
                                            lens.source_layers.index(layer),
                                        ],
                                        layer,
                                    )
                                )
                                trained_rank = _rank_percentile(trained_logits, ids).mean(dim=-1)
                                base_rank = _rank_percentile(base_logits, ids).mean(dim=-1)
                                stimulus_diffs = trained_rank - base_rank
                                for stimulus_index, manifest_index in enumerate(indices):
                                    pair_values.setdefault(
                                        str(manifests[manifest_index]["pair_id"]), []
                                    ).append(float(stimulus_diffs[stimulus_index]))
                                trained_scores.append(float(trained_rank.mean()))
                                base_scores.append(float(base_rank.mean()))
                                trained_logits_means.append(float(trained_logits[:, ids].mean()))
                                base_logits_means.append(float(base_logits[:, ids].mean()))
                                profile.append(
                                    {
                                        "layer": float(layer),
                                        "trained_rank_percentile": trained_scores[-1],
                                        "base_rank_percentile": base_scores[-1],
                                        "rank_diff": trained_scores[-1] - base_scores[-1],
                                        "trained_mean_logit": trained_logits_means[-1],
                                        "base_mean_logit": base_logits_means[-1],
                                    }
                                )
                            pair_means = [
                                sum(values) / len(values) for values in pair_values.values()
                            ]
                            concept_report[concept_name] = {
                                "trained_band_mean": sum(trained_scores) / len(trained_scores),
                                "base_band_mean": sum(base_scores) / len(base_scores),
                                "diff": sum(trained_scores) / len(trained_scores)
                                - sum(base_scores) / len(base_scores),
                                "trained_mean_logit": sum(trained_logits_means)
                                / len(trained_logits_means),
                                "base_mean_logit": sum(base_logits_means) / len(base_logits_means),
                                "mean_logit_diff": sum(trained_logits_means)
                                / len(trained_logits_means)
                                - sum(base_logits_means) / len(base_logits_means),
                                "bootstrap_ci": list(_bootstrap_ci(pair_means, seed=args.seed)),
                                "profile": profile,
                            }
                        report["concept_sets"][f"{arm}|{side}|{position_name}"] = concept_report
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mean_logits, args.out_dir / "mean_logits.pt")
    (args.out_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Workspace readout",
        "",
        f"Band: layers {band_start}:{band_end}",
        "",
        "The vocabulary filter keeps decoded tokens of length at least three containing a letter or CJK character.",
        "",
    ]
    for key, values in report["top_tokens"].items():
        lines.append(f"## {key}")
        lines.extend(f"- {row['token']!r}: {row['logit']:.5f}" for row in values)
    for key, values in report["paired_differences"].items():
        lines.extend(("", f"## Paired differences: {key}", "### Risen"))
        lines.extend(
            f"- {row['token']!r}: Δ={row['delta']:.5f}, fraction_positive={row['fraction_positive']:.3f}"
            for row in values["risen"]
        )
        lines.append("### Fallen")
        lines.extend(
            f"- {row['token']!r}: Δ={row['delta']:.5f}, fraction_positive={row['fraction_positive']:.3f}"
            for row in values["fallen"]
        )
    for key, values in report["concept_sets"].items():
        lines.extend(("", f"## Concept sets: {key}"))
        lines.extend(
            f"- {name}: rank diff={entry['diff']:.5f}, bootstrap CI={entry['bootstrap_ci']}"
            for name, entry in values.items()
        )
    (args.out_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def word_variant_token_ids(tokenizer: PreTrainedTokenizerBase, word: str) -> list[int]:
    """Single-token ids of a word as it can appear mid-text: bare or space-led, as given or capitalised."""
    variants = {word, f" {word}", word.capitalize(), f" {word.capitalize()}"}
    ids = {tuple(_token_ids(tokenizer, variant)) for variant in variants}
    return sorted(single[0] for single in ids if len(single) == 1)


def sanity(args: argparse.Namespace) -> None:
    """Run the runtime-only multihop lens-quality evaluation."""
    evaluations = cast("dict[str, Any]", json.loads(args.evaluations.read_text(encoding="utf-8")))
    items = evaluations["items"]
    lens = load_lens(args.lens)
    jlens = _load_jlens()
    transformers = __import__("transformers", fromlist=["AutoModelForCausalLM", "AutoTokenizer"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(
        args.device
    )
    jl_model = jlens.from_hf(model, tokenizer)
    reference_lens = jlens.JacobianLens.load(str(args.lens))
    pass_rates: dict[str, dict[str, float]] = {
        kind: {str(k): 0.0 for k in (1, 5, 10, 25)} for kind in ("jacobian", "logit_lens")
    }
    layer_hits: dict[str, dict[str, dict[str, int]]] = {kind: {} for kind in pass_rates}
    layer_totals: dict[str, dict[str, int]] = {kind: {} for kind in pass_rates}
    for item in items:
        prompt = str(item["prompt"])
        target = str(item["target"])
        combined = prompt + target
        prompt_ids = jl_model.encode(prompt, max_length=4096)
        position = int(prompt_ids.shape[-1]) - 1
        j_logits, _, _ = reference_lens.apply(
            jl_model,
            combined,
            layers=list(lens.source_layers),
            positions=[position],
            use_jacobian=True,
        )
        l_logits, _, _ = reference_lens.apply(
            jl_model,
            combined,
            layers=list(lens.source_layers),
            positions=[position],
            use_jacobian=False,
        )
        word_variant_ids = [
            ids
            for ids in (
                word_variant_token_ids(tokenizer, str(word)) for word in item["intermediates"]
            )
            if ids
        ]
        for kind, logits_by_layer in (("jacobian", j_logits), ("logit_lens", l_logits)):
            # rank of each intermediate at each layer: best rank over its single-token variants
            ranks_by_word = [
                [
                    min(int((logits[0] > logits[0, token_id]).sum()) for token_id in ids)
                    for logits in logits_by_layer.values()
                ]
                for ids in word_variant_ids
            ]
            for k in (1, 5, 10, 25):
                passed = sum(min(ranks) < k for ranks in ranks_by_word)
                pass_rates[kind][str(k)] += passed / len(ranks_by_word) if ranks_by_word else 0.0
            for layer_index, layer in enumerate(logits_by_layer):
                row = layer_hits[kind].setdefault(str(layer), {str(k): 0 for k in (1, 5, 10, 25)})
                layer_totals[kind][str(layer)] = layer_totals[kind].get(str(layer), 0) + len(
                    ranks_by_word
                )
                for k in (1, 5, 10, 25):
                    row[str(k)] += sum(ranks[layer_index] < k for ranks in ranks_by_word)
    for rates in pass_rates.values():
        for k in rates:
            rates[k] /= len(items)
    payload = {
        "normalized_pass_at_k": pass_rates,
        "items": len(items),
        "per_layer_hits": layer_hits,
        "per_layer_fraction": {
            kind: {
                layer: {
                    key: value / max(1, layer_totals[kind][layer]) for key, value in counts.items()
                }
                for layer, counts in layer_hits[kind].items()
            }
            for kind in layer_hits
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sanity.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "sanity.md").write_text(
        "# Lens sanity\n\n" + json.dumps(payload["normalized_pass_at_k"], indent=2) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--lens", type=Path, required=True)
    capture_parser.add_argument("--base-model", required=True)
    capture_parser.add_argument("--out-dir", type=Path, required=True)
    capture_parser.add_argument("--arm", action="append", default=[])
    capture_parser.add_argument("--framing", dest="framings", action="append", default=None)
    capture_parser.add_argument("--device", default="cuda")
    capture_parser.add_argument("--target-layer", type=int, default=30)
    capture_parser.add_argument(
        "--base-model-alias",
        action="append",
        default=[],
        help="another id for the SAME base weights (e.g. the hub id of a pinned local snapshot) that "
        "an adapter may record as its base; recorded in run.json",
    )
    analyse_parser = subparsers.add_parser("analyse")
    analyse_parser.add_argument("--capture-dir", type=Path, required=True)
    analyse_parser.add_argument("--lens", type=Path, required=True)
    analyse_parser.add_argument("--model-path", type=Path, required=True)
    analyse_parser.add_argument("--out-dir", type=Path, required=True)
    analyse_parser.add_argument("--band", default="10:26")
    analyse_parser.add_argument("--concept-sets", type=Path)
    analyse_parser.add_argument("--seed", type=int, default=20260926)
    analyse_parser.add_argument("--device", default="cpu")
    sanity_parser = subparsers.add_parser("sanity")
    sanity_parser.add_argument("--lens", type=Path, required=True)
    sanity_parser.add_argument("--model", required=True)
    sanity_parser.add_argument("--evaluations", type=Path, default=Path(LENS_EVAL_PATH))
    sanity_parser.add_argument("--out-dir", type=Path, required=True)
    sanity_parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one of the capture, analyse, or sanity subcommands."""
    args = _parser().parse_args(argv)
    if args.command == "capture":
        args.framings = tuple(args.framings or DEFAULT_FRAMINGS)
        capture(args)
    elif args.command == "analyse":
        analyse(args)
    else:
        sanity(args)


if __name__ == "__main__":
    main()
