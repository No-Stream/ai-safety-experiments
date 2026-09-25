"""Inference-only argument-prior and truncated-donor map."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer

from games.eval_model import read_adapter_facts
from games.evals import EVAL_RENDER_GRADING_BY_GAME
from games.inference_utils import (
    adapter_digest,
    append_jsonl,
    content_key,
    derive_batch_size,
    device_memory_report,
    ensure_run_identity,
    load_completed_keys,
    resolve_device,
)
from games.interp_mediation import (
    assert_clean_boundary,
    continue_raw,
    render_context,
    thinking_segment,
)
from games.lora import (
    ADAPTER_CONFIG_FILENAME,
    assert_one_adapter_config,
    attach_adapter,
    load_adapter_base,
)
from games.parsing import THINK_CLOSE, parse_action, strip_thinking
from games.payoffs import COOPERATE
from games.prompts import (
    LABEL_PRINT_ORDERS,
    generate_framing_prompt_rows,
)
from games.teacher_forcing import (
    TeacherForcingTokenizer,
    build_batch,
    continuation_logprob_means,
    continuation_logprob_sums,
    teacher_force,
)
from games.trace_judge_schema import MIRROR_RE
from games.vllm_teardown import release_engine, vram_used_mib
from reward_hacking.model_backend import SamplingConfig, VLLMBackend

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_MATRIX_DIMENSIONS = 2

BASE_MODEL = "Qwen/Qwen3.5-9B"
SMOKE_MODEL = "Qwen/Qwen3.5-0.8B"
SELF_ADAPTER = Path("artifacts/games/ninep-transfer-2026-09-23/adapters/twin-pd-self-step70")
GROUP_ADAPTER = Path("artifacts/games/ninep-transfer-2026-09-23/adapters/twin-pd-group-step70")
DEFAULT_STIMULUS = Path("docs/scratch/rl-generalization-inference/argument-prior-openers.json")
DEFAULT_RECORDS_ROOT = Path("artifacts/games/ninep-transfer-2026-09-23/records/framing")
DEFAULT_OUTPUT = Path("artifacts/games/argument-prior-map/first-cut")
GAMES: tuple[str, ...] = ("twin-pd", "public-goods")
# The first-cut map deliberately excludes the registered dose/control extensions.  Keeping this
# roster local makes a future expansion an explicit design change rather than an accidental registry
# change upstream.
FRAMINGS: tuple[str, ...] = (
    "twin",
    "same-weights-uncorrelated",
    "another-ai",
    "different-ai",
    "different-ai-coupled",
    "human",
    "stated-matcher",
    "stated-track-record",
    "unstated",
)
MODEL_CONDITIONS: tuple[str, ...] = ("base", "self", "group")
COMMITMENT_RE = re.compile(
    r"\b(?:i(?:'m| am)\s+going\s+to\s+|i(?:'ll| will| should)\s+"
    r"(?:choose|pick|select|play|do)\s+|(?:choose|pick|select|play)\s+)"
    r"(?P<label>[A-Za-z][A-Za-z0-9_-]*)\b|"
    r"\b(?:i(?:'ll| will| should)\s+)(?:cooperate|defect)\b",
    re.IGNORECASE,
)

# Donor continuations per vLLM call: the engine batches a chunk, and records land per chunk so a
# killed run loses at most one chunk.
DONOR_VLLM_CHUNK = 64


def _read_safetensor_shapes(adapter_dir: Path) -> dict[str, tuple[int, ...]]:
    """Read adapter tensor shapes without loading the weights into the inference process."""
    safetensor_path = adapter_dir / "adapter_model.safetensors"
    if not safetensor_path.is_file():
        raise ValueError(
            f"cannot validate adapter shapes in {adapter_dir}: only safetensors adapters are "
            "supported by the vLLM preflight"
        )
    with safe_open(str(safetensor_path), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        return {key: tuple(handle.get_slice(key).get_shape()) for key in keys}


def _assert_adapter_shapes_match_model(adapter_dir: Path, model: nn.Module) -> None:
    """Refuse LoRA tensors whose dimensions do not match the loaded model's target modules."""
    shapes = _read_safetensor_shapes(adapter_dir)
    config_path = adapter_dir / ADAPTER_CONFIG_FILENAME
    config = cast("dict[str, Any]", json.loads(config_path.read_text(encoding="utf-8")))
    rank = config.get("r")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError(f"{config_path} has no positive integer LoRA rank")
    suffixes = (".lora_A.weight", ".lora_B.weight")
    checked = 0
    for key, shape in shapes.items():
        suffix = next((candidate for candidate in suffixes if key.endswith(candidate)), None)
        if suffix is None:
            continue
        module_name = key[: -len(suffix)].removeprefix("base_model.model.")
        try:
            module = model.get_submodule(module_name)
        except AttributeError as error:
            raise ValueError(
                f"Adapter {adapter_dir} targets missing model module {module_name!r}; its "
                "module tree does not match the selected base"
            ) from error
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != _MATRIX_DIMENSIONS:
            raise ValueError(
                f"Adapter {adapter_dir} targets {module_name!r}, which has no 2-D weight "
                "for the LoRA shape check"
            )
        expected = (
            (rank, int(weight.shape[1])) if suffix == suffixes[0] else (int(weight.shape[0]), rank)
        )
        if shape != expected:
            raise ValueError(
                f"Adapter {adapter_dir} tensor {key!r} has shape {shape}, expected {expected} "
                f"for base module {module_name!r}; refusing to start vLLM"
            )
        checked += 1
    if checked == 0:
        raise ValueError(f"Adapter {adapter_dir} contains no LoRA A/B tensors to shape-check")


def _assert_vllm_adapter_preflight(adapter_dir: Path, model_id: str) -> None:
    """Check adapter identity and internal rank shapes before constructing a vLLM engine."""
    config_path = adapter_dir / ADAPTER_CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} not found; refusing to start vLLM")
    config = cast("dict[str, Any]", json.loads(config_path.read_text(encoding="utf-8")))
    recorded = str(config.get("base_model_name_or_path", "")).rstrip("/")
    expected = model_id.rstrip("/")
    if not recorded:
        raise ValueError(f"{config_path} has no base_model_name_or_path; refusing to start vLLM")
    if recorded != expected:
        raise ValueError(
            f"Adapter {adapter_dir} was trained against {recorded!r}, but vLLM was asked to "
            f"serve {expected!r}; the adapter/base mismatch must be fixed before engine startup"
        )
    shapes = _read_safetensor_shapes(adapter_dir)
    rank = config.get("r")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError(f"{config_path} has no positive integer LoRA rank; refusing to start vLLM")
    mismatched = [
        (key, shape)
        for key, shape in shapes.items()
        if (
            key.endswith(".lora_A.weight")
            and (len(shape) != _MATRIX_DIMENSIONS or shape[0] != rank)
        )
        or (
            key.endswith(".lora_B.weight")
            and (len(shape) != _MATRIX_DIMENSIONS or shape[1] != rank)
        )
    ]
    if mismatched:
        key, shape = mismatched[0]
        raise ValueError(
            f"Adapter {adapter_dir} has tensor {key!r} with shape {shape}, inconsistent with "
            f"configured LoRA rank {rank}; refusing to start vLLM"
        )


def _randomize_lora_weights(model: nn.Module, *, seed: int) -> None:
    """Give a smoke adapter non-zero deterministic A and B weights."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)
            if lora_a is None or lora_b is None:
                continue
            for adapter_name in lora_a:
                a_weight = lora_a[adapter_name].weight
                b_weight = lora_b[adapter_name].weight
                a_random = torch.randn(
                    a_weight.shape, generator=generator, dtype=torch.float32, device="cpu"
                ).to(device=a_weight.device, dtype=a_weight.dtype)
                b_random = torch.randn(
                    b_weight.shape, generator=generator, dtype=torch.float32, device="cpu"
                ).to(device=b_weight.device, dtype=b_weight.dtype)
                a_weight.copy_(a_random)
                b_weight.copy_(b_random)


def _build_smoke_adapters(
    base_model: PreTrainedModel,
    *,
    source_adapters: tuple[Path, Path],
    output_root: Path,
    model_id: str,
) -> tuple[PreTrainedModel, tuple[Path, Path]]:
    """Build 0.8B LoRAs with the production target list for the vLLM switching smoke."""
    output_paths: tuple[Path, Path] = (output_root / "self", output_root / "group")
    current_model = base_model
    for index, (source_adapter, output_path) in enumerate(
        zip(source_adapters, output_paths, strict=True)
    ):
        source_config = cast(
            "dict[str, Any]",
            json.loads((source_adapter / ADAPTER_CONFIG_FILENAME).read_text(encoding="utf-8")),
        )
        target_modules = source_config.get("target_modules")
        if not isinstance(target_modules, list) or not target_modules:
            raise ValueError(
                f"{source_adapter / ADAPTER_CONFIG_FILENAME} has no target_modules for smoke LoRA"
            )
        smoke_model = get_peft_model(
            current_model,
            LoraConfig(
                r=4,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[str(module) for module in target_modules],
            ),
        )
        _randomize_lora_weights(smoke_model, seed=20260924 + index)
        smoke_model.peft_config["default"].base_model_name_or_path = model_id
        output_path.mkdir(parents=True, exist_ok=True)
        smoke_model.save_pretrained(str(output_path), safe_serialization=True)
        current_model = cast("PreTrainedModel", cast("Any", smoke_model).unload())
    return current_model, output_paths


@dataclass(frozen=True, slots=True)
class Opener:
    """One authored opener loaded from the private stimulus file."""

    identifier: str
    category: str
    text: str


@dataclass(frozen=True, slots=True)
class ContextRow:
    """A regenerated prompt row and its rendered thinking-open context."""

    game_id: str
    framing_id: str
    prompt_id: str
    label_print_order: str
    prompt: str
    context: str
    label_a: str
    label_b: str
    coop_label: str


@dataclass(frozen=True, slots=True)
class Donor:
    """A banked step-0 trace selected for truncated continuation."""

    game_id: str
    framing_id: str
    prompt_id: str
    sample_index: int
    label_print_order: str
    completion: str
    thinking: str
    label_a: str
    label_b: str
    coop_label: str


def _json_digest(path: Path) -> str:
    """Hash a stimulus file so a changed private bank cannot silently resume an old run."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_openers(path: Path) -> tuple[Opener, ...]:
    """Load and validate the opener bank without embedding its item text in tracked code."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_openers = payload.get("openers")
    if not isinstance(raw_openers, list) or not raw_openers:
        raise ValueError(f"{path} has no non-empty 'openers' list")
    openers = tuple(
        Opener(str(item["id"]), str(item["category"]), str(item["text"])) for item in raw_openers
    )
    if len({opener.identifier for opener in openers}) != len(openers):
        raise ValueError(f"{path} contains duplicate opener ids")
    if any(not opener.text.strip() for opener in openers):
        raise ValueError(f"{path} contains an empty opener")
    categories = {opener.category for opener in openers}
    required_categories = {"mirror", "dominance", "neutral"}
    if not required_categories <= categories:
        raise ValueError(
            f"{path} must contain mirror, dominance, and neutral opener categories; found {categories}"
        )
    return openers


def spread_across_scenarios(
    rendered_rows: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Pick `count` rows round-robin across scenarios, rotating through each one's label arrangements.

    Sorting by prompt id and taking the first rows once gave every cell a single scenario in four
    label arrangements, so the spread measured label order rather than scenario variation.
    """
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rendered_rows:
        by_scenario[str(row["reskin_id"])].append(row)
    scenarios = [
        sorted(rows, key=lambda item: (str(item["prompt_id"]), str(item["label_print_order"])))
        for _scenario, rows in sorted(by_scenario.items())
    ]
    picked: list[dict[str, Any]] = []
    round_index = 0
    while len(picked) < count and any(round_index < len(rows) for rows in scenarios):
        for scenario_index, rows in enumerate(scenarios):
            arrangement = (round_index + scenario_index) % len(rows)
            if round_index < len(rows) and len(picked) < count:
                picked.append(rows[arrangement])
        round_index += 1
    return picked


def context_rows(
    tokenizer: TeacherForcingTokenizer,
    *,
    rows_per_cell: int,
    smoke: bool = False,
) -> tuple[ContextRow, ...]:
    """Regenerate a deterministic subset of every requested game/framing cell."""
    if rows_per_cell < 1:
        raise ValueError("rows_per_cell must be positive")
    selected: list[ContextRow] = []
    framing_ids = ("twin", "human") if smoke else FRAMINGS
    game_ids = ("twin-pd",) if smoke else GAMES
    for game_id in game_ids:
        for framing_id in framing_ids:
            rendered_rows = [
                row
                for label_print_order in LABEL_PRINT_ORDERS
                for row in generate_framing_prompt_rows(
                    game_id,
                    EVAL_RENDER_GRADING_BY_GAME[game_id],
                    framing_id=framing_id,
                    split="eval",
                    label_print_order=label_print_order,
                )
            ]
            for row in spread_across_scenarios(rendered_rows, rows_per_cell):
                prompt = str(row["prompt"])
                selected.append(
                    ContextRow(
                        game_id=game_id,
                        framing_id=framing_id,
                        prompt_id=str(row["prompt_id"]),
                        label_print_order=str(row["label_print_order"]),
                        prompt=prompt,
                        context=render_context(tokenizer, prompt),
                        label_a=str(row["label_a"]),
                        label_b=str(row["label_b"]),
                        coop_label=str(row["coop_label"]),
                    )
                )
    return tuple(selected)


def _hidden_size(model: PreTrainedModel) -> int:
    """Read the text tower width used for activation-size batch derivation."""
    config = cast("Any", model.config)
    text_config = getattr(config, "text_config", config)
    return int(text_config.hidden_size)


def _score_openers(  # noqa: PLR0913 - batch scorer takes one argument per measurement axis
    model: nn.Module,
    tokenizer: TeacherForcingTokenizer,
    rows: Sequence[ContextRow],
    openers: Sequence[Opener],
    *,
    batch_size: int,
    disable_adapter: bool,
) -> Iterator[tuple[ContextRow, Opener, float, float, int]]:
    """Score all opener continuations in bounded teacher-forcing batches."""
    units = [(row, opener) for row in rows for opener in openers]
    for start in range(0, len(units), batch_size):
        chunk = units[start : start + batch_size]
        contexts = [row.context for row, _ in chunk]
        continuations = [opener.text for _, opener in chunk]
        batch = build_batch(tokenizer, contexts, continuations)
        scope = (
            cast("Any", model).disable_adapter()
            if disable_adapter and hasattr(model, "disable_adapter")
            else nullcontext()
        )
        with scope:
            result = teacher_force(model, batch)
        sums = continuation_logprob_sums(result)
        means = continuation_logprob_means(result)
        for (row, opener), total, mean, token_ids in zip(
            chunk, sums, means, result.continuation_lengths, strict=True
        ):
            yield row, opener, total, mean, token_ids


def _read_bank(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read one banked JSONL cell, keeping its metadata separate from records."""
    meta: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            if row.get("record") == "meta":
                meta = row
            else:
                records.append(row)
    if meta is None:
        raise ValueError(f"{path} has no meta row")
    return meta, records


def behavior_summary(  # noqa: C901, PLR0912 - the four bank cells and two summaries are explicit
    records_root: Path, *, strict: bool = True
) -> dict[str, Any]:
    """Compute free cooperation and base mirror-invocation columns from banked records."""
    rates: dict[str, dict[str, int]] = defaultdict(lambda: {"cooperate": 0, "parsed": 0})
    mirror: dict[str, dict[str, int]] = defaultdict(lambda: {"mirror": 0, "records": 0})
    self_step0 = records_root / "9b-twin-pd-self" / "step-0.jsonl"
    missing: list[Path] = []
    for arm in ("self", "group"):
        for step in (0, 70):
            path = records_root / f"9b-twin-pd-{arm}" / f"step-{step}.jsonl"
            if not path.is_file():
                missing.append(path)
                continue
            _, records = _read_bank(path)
            for row in records:
                if row.get("record") != "framing-sweep":
                    continue
                game_id = str(row.get("game_id"))
                framing = str(row.get("counterpart_framing"))
                key = f"{game_id}|{framing}|step-{step}|{arm}"
                if bool(row.get("parsed")):
                    rates[key]["parsed"] += 1
                    if row.get("coop_fraction") is not None and float(row["coop_fraction"]) == 1.0:
                        rates[key]["cooperate"] += 1
    if missing and strict:
        raise FileNotFoundError("missing banked behavior files: " + ", ".join(map(str, missing)))
    for path in missing:
        logger.warning("banked behavior file missing: %s", path)
    if self_step0.is_file():
        _, records = _read_bank(self_step0)
        for row in records:
            if row.get("record") != "framing-sweep":
                continue
            key = f"{row.get('game_id')}|{row.get('counterpart_framing')}"
            mirror[key]["records"] += 1
            if MIRROR_RE.search(str(row.get("completion", ""))):
                mirror[key]["mirror"] += 1
    return {
        "cooperation": {
            key: {
                **counts,
                "rate": counts["cooperate"] / counts["parsed"] if counts["parsed"] else None,
            }
            for key, counts in sorted(rates.items())
        },
        "base_step0_mirror_invocation": {
            key: {
                **counts,
                "rate": counts["mirror"] / counts["records"] if counts["records"] else None,
            }
            for key, counts in sorted(mirror.items())
        },
    }


def _commitment_end(thinking: str, labels: tuple[str, str]) -> int | None:
    """Return the first sentence boundary strictly before an action commitment."""
    label_pattern = re.compile(
        rf"\b(?:{re.escape(labels[0])}|{re.escape(labels[1])})\b", re.IGNORECASE
    )
    candidates = [
        match
        for match in COMMITMENT_RE.finditer(thinking)
        if label_pattern.search(match.group(0))
        or re.search(r"cooperate|defect", match.group(0), re.IGNORECASE)
    ]
    if not candidates:
        return None
    commitment_start = candidates[0].start()
    boundaries = [match.end() for match in re.finditer(r"[.!?](?:\s+|$)|\n+", thinking)]
    before = [boundary for boundary in boundaries if boundary <= commitment_start]
    return max(before, default=0)


def _cut_at_sentence_boundaries(text: str, fraction: float, end: int) -> str:
    """Cut the pre-commitment span at the latest sentence boundary below a target fraction."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("cut fraction must lie in [0, 1]")
    target = int(end * fraction)
    boundaries = [match.end() for match in re.finditer(r"[.!?](?:\s+|$)|\n+", text[:end])]
    if fraction == 1.0:
        return text[:end]
    valid = [boundary for boundary in boundaries if boundary <= target]
    return text[: max(valid, default=0)]


def _select_donors(  # noqa: C901, PLR0912 - selection guards preserve donor provenance
    records_root: Path,
    *,
    donor_count: int,
    smoke: bool,
) -> tuple[Donor, ...]:
    """Select deterministic twin/human step-0 donors that reach an answer."""
    candidates_by_framing: dict[str, list[Donor]] = {"twin": [], "human": []}
    path = records_root / "9b-twin-pd-self" / "step-0.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"donor bank is missing: {path}")
    _, records = _read_bank(path)
    for row in records:
        framing = str(row.get("counterpart_framing"))
        if (
            str(row.get("game_id")) != "twin-pd"
            or framing not in candidates_by_framing
            or not bool(row.get("parsed"))
        ):
            continue
        completion = str(row.get("completion", ""))
        if THINK_CLOSE not in completion:
            continue
        try:
            thinking = thinking_segment(completion)
        except ValueError:
            continue
        labels = (str(row["label_a"]), str(row["label_b"]))
        if _commitment_end(thinking, labels) is None:
            continue
        candidates_by_framing[framing].append(
            Donor(
                game_id=str(row["game_id"]),
                framing_id=framing,
                prompt_id=str(row["prompt_id"]),
                sample_index=int(row.get("sample_index", 0)),
                label_print_order=str(row.get("label_print_order", "canonical")),
                completion=completion,
                thinking=thinking,
                label_a=labels[0],
                label_b=labels[1],
                coop_label=str(row["coop_label"]),
            )
        )
    for framing, candidates in candidates_by_framing.items():
        candidates_by_framing[framing] = sorted(
            candidates,
            key=lambda donor: (donor.prompt_id, donor.sample_index, donor.completion),
        )
    selected: list[Donor] = []
    seen: set[str] = set()
    while len(selected) < donor_count and any(candidates_by_framing.values()):
        for framing in ("twin", "human"):
            if len(selected) >= donor_count:
                break
            if not candidates_by_framing[framing]:
                continue
            donor = candidates_by_framing[framing].pop(0)
            donor_identity = content_key(donor.prompt_id, donor.sample_index, donor.completion)
            if donor_identity in seen:
                continue
            seen.add(donor_identity)
            selected.append(donor)
    if smoke:
        selected = selected[:1]
    return tuple(selected[:donor_count])


def _raw_hf_continue(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    contexts: Sequence[str],
    *,
    max_new_tokens: int,
) -> list[str]:
    """CPU smoke fallback: continue raw rendered contexts without applying another chat template."""
    outputs: list[str] = []
    for context in contexts:
        encoded = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        encoded = encoded.to(next(model.parameters()).device)
        generate = cast("Any", model).generate
        generated = generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
        )
        prompt_length = int(encoded["input_ids"].shape[1])
        outputs.append(
            cast("str", tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=False))
        )
    return outputs


def donor_cooperation_rates(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Cooperation per (donor framing, cut, model), counted from the parsed canonical action."""
    grouped: dict[tuple[str, float, str], dict[str, int]] = defaultdict(
        lambda: {"records": 0, "parsed": 0, "cooperate": 0}
    )
    for record in records:
        counts = grouped[
            (
                str(record["donor_framing"]),
                float(record["cut_fraction"]),
                str(record["model_condition"]),
            )
        ]
        counts["records"] += 1
        if record.get("parsed_action") is not None:
            counts["parsed"] += 1
            counts["cooperate"] += int(record["parsed_action"] == COOPERATE)
    return {
        f"{framing}|cut-{cut:g}|{condition}": {
            **counts,
            "rate": counts["cooperate"] / counts["parsed"] if counts["parsed"] else None,
        }
        for (framing, cut, condition), counts in sorted(grouped.items())
    }


def _run_donor_part(  # noqa: C901, PLR0912, PLR0913, PLR0915
    args: argparse.Namespace,
    base_model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizerBase,
    *,
    rows: Sequence[ContextRow],
    attached: Any | None,  # noqa: ANN401
    output_path: Path,
    stimulus_digest: str,
    device: torch.device,
) -> dict[str, Any]:
    """Run truncated-donor continuations, using raw vLLM in production and HF on CPU smoke."""
    donors = _select_donors(
        args.records_root,
        donor_count=args.donors,
        smoke=args.smoke,
    )
    row_by_prompt = {(row.prompt_id, row.label_print_order): row for row in rows}
    completed = load_completed_keys(output_path)
    donor_bank_digest = _json_digest(args.records_root / "9b-twin-pd-self" / "step-0.jsonl")
    resumed = 0
    written = 0
    cuts = (0.0, 1.0) if args.smoke else (0.0, 1 / 3, 2 / 3, 1.0)
    samples = 1 if args.smoke else args.samples
    work: list[tuple[Donor, ContextRow, float, str, str, int]] = []
    for donor in donors:
        row = row_by_prompt.get((donor.prompt_id, donor.label_print_order))
        if row is None:
            for label_print_order in LABEL_PRINT_ORDERS:
                generated = generate_framing_prompt_rows(
                    donor.game_id,
                    EVAL_RENDER_GRADING_BY_GAME[donor.game_id],
                    framing_id=donor.framing_id,
                    split="eval",
                    label_print_order=label_print_order,
                )
                match = next(
                    (
                        candidate
                        for candidate in generated
                        if candidate["prompt_id"] == donor.prompt_id
                    ),
                    None,
                )
                if match is not None:
                    row = ContextRow(
                        game_id=donor.game_id,
                        framing_id=donor.framing_id,
                        prompt_id=str(match["prompt_id"]),
                        label_print_order=str(match["label_print_order"]),
                        prompt=str(match["prompt"]),
                        context=render_context(cast("Any", tokenizer), str(match["prompt"])),
                        label_a=str(match["label_a"]),
                        label_b=str(match["label_b"]),
                        coop_label=str(match["coop_label"]),
                    )
                    break
        if row is None:
            raise ValueError(f"donor prompt {donor.prompt_id} was not regenerated")
        end = _commitment_end(donor.thinking, (donor.label_a, donor.label_b))
        if end is None:
            continue
        for cut_fraction in cuts:
            transplant = _cut_at_sentence_boundaries(donor.thinking, cut_fraction, end)
            if transplant:
                if not donor.completion.startswith(transplant):
                    raise ValueError(
                        f"donor cut for {donor.prompt_id!r} is not a byte-exact completion prefix"
                    )
                assert_clean_boundary(cast("Any", tokenizer), row.context, transplant)
            work.extend(
                (donor, row, cut_fraction, transplant, row.context + transplant, sample_index)
                for sample_index in range(samples)
            )

    for condition in MODEL_CONDITIONS:
        backend: VLLMBackend | None = None
        baseline_mib: list[int] | None = None
        if base_model is None and device.type == "cpu":
            raise RuntimeError("CPU donor continuation requires the loaded base model")
        active_model: Any = base_model if attached is None else attached.peft_model
        if device.type == "cpu":
            if condition == "self" and attached is not None:
                attached = attach_adapter(
                    cast("PreTrainedModel", base_model),
                    args.self_adapter,
                    args.model,
                    existing=active_model,
                )
                active_model = attached.peft_model
            elif condition == "group" and attached is not None:
                attached = attach_adapter(
                    cast("PreTrainedModel", base_model),
                    args.group_adapter,
                    args.model,
                    existing=active_model,
                )
                active_model = attached.peft_model
        else:
            sampling = SamplingConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=1.0,
                top_p=0.95,
                top_k=0,
            )
            adapter: Path | None = None
            kwargs: dict[str, Any] = {"language_model_only": True}
            if condition == "self":
                adapter = args.self_adapter
            elif condition == "group":
                adapter = args.group_adapter
            if adapter is not None:
                _assert_vllm_adapter_preflight(adapter, args.model)
                facts = read_adapter_facts(adapter)
                kwargs.update(
                    {
                        "enable_lora": True,
                        "max_lora_rank": facts.vllm_lora_rank,
                        "lora_target_modules": list(facts.target_modules),
                    }
                )
            baseline_mib = vram_used_mib()
            backend = VLLMBackend(
                args.model,
                thinking=True,
                sampling=sampling,
                lora_adapter=adapter,
                model_path=args.model_path,
                **kwargs,
            )
            if adapter is not None and work:
                backend.assert_adapter_changes_output([work[0][1].prompt])
        try:
            pending: list[tuple[str, Donor, float, str, str, int]] = []
            for donor, _row, cut_fraction, transplant, raw_context, sample_index in work:
                key = content_key(
                    "donor",
                    stimulus_digest,
                    donor_bank_digest,
                    args.model,
                    str(args.model_path or ""),
                    donor.prompt_id,
                    donor.framing_id,
                    donor.sample_index,
                    donor.completion,
                    cut_fraction,
                    transplant,
                    raw_context,
                    condition,
                    args.adapter_digests.get(condition, ""),
                    sample_index,
                )
                if key in completed:
                    resumed += 1
                else:
                    pending.append(
                        (key, donor, cut_fraction, transplant, raw_context, sample_index)
                    )
            chunk_size = DONOR_VLLM_CHUNK if backend is not None else 1
            for start in range(0, len(pending), chunk_size):
                chunk = pending[start : start + chunk_size]
                contexts = [
                    raw_context for _key, _donor, _cut, _transplant, raw_context, _i in chunk
                ]
                if backend is not None:
                    texts = [continuation.text for continuation in continue_raw(backend, contexts)]
                else:
                    scope = (
                        active_model.disable_adapter()
                        if condition == "base" and attached is not None
                        else nullcontext()
                    )
                    with scope:
                        texts = _raw_hf_continue(
                            active_model,
                            tokenizer,
                            contexts,
                            max_new_tokens=args.max_new_tokens,
                        )
                for (key, donor, cut_fraction, transplant, _context, sample_index), text in zip(
                    chunk, texts, strict=True
                ):
                    visible, truncated = strip_thinking(text, prefilled_think=True)
                    action = parse_action(
                        visible,
                        label_a=donor.label_a,
                        label_b=donor.label_b,
                        coop_label=donor.coop_label,
                    )
                    append_jsonl(
                        output_path,
                        {
                            "key": key,
                            "record": "donor_continuation",
                            "donor_prompt_id": donor.prompt_id,
                            "donor_framing": donor.framing_id,
                            "cut_fraction": cut_fraction,
                            "cut_text": transplant,
                            "model_condition": condition,
                            "adapter_available": condition != "base",
                            "sample_index": sample_index,
                            "completion": text,
                            "parsed_action": action,
                            "cooperated": action == COOPERATE,
                            "truncated_thinking": truncated,
                            "device": str(device),
                        },
                    )
                    completed.add(key)
                    written += 1
        finally:
            if backend is not None and baseline_mib is not None:
                release_engine(backend, baseline_mib=baseline_mib)
    output_records = (
        [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if output_path.exists()
        else []
    )
    cooperation_rates = donor_cooperation_rates(output_records)
    return {
        "written": written,
        "resumed": resumed,
        "donors": len(donors),
        "cooperation_rates": cooperation_rates,
    }


def run_argument_prior(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915
    """Run the two argument-prior parts and write incremental artifacts."""
    if args.smoke and args.model not in (None, SMOKE_MODEL):
        raise ValueError(f"--smoke requires --model {SMOKE_MODEL!r} when --model is supplied")
    args.model = SMOKE_MODEL if args.smoke else (args.model or BASE_MODEL)
    args.model_path = None if args.model_path is None else Path(args.model_path)
    args.self_adapter = Path(args.self_adapter)
    args.group_adapter = Path(args.group_adapter)
    args.stimulus = Path(args.stimulus)
    args.output = Path(args.output)
    args.records_root = Path(args.records_root)
    args.output.mkdir(parents=True, exist_ok=True)
    args.adapter_digests = (
        {
            "self": adapter_digest(args.self_adapter),
            "group": adapter_digest(args.group_adapter),
        }
        if not args.smoke
        else {}
    )
    stimulus_digest = _json_digest(args.stimulus)
    ensure_run_identity(
        args.output / "run_identity.json",
        {
            "experiment": "argument-prior-map",
            "model": args.model,
            "model_path": str(args.model_path or ""),
            "adapter_digests": args.adapter_digests,
            "stimulus_sha256": stimulus_digest,
        },
    )
    device = resolve_device(args.device)
    memory = device_memory_report(device)
    model_source = str(args.model_path) if args.model_path is not None else str(args.model)
    tokenizer = cast("PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(model_source))
    rows = context_rows(cast("Any", tokenizer), rows_per_cell=args.rows_per_cell, smoke=args.smoke)
    openers = load_openers(args.stimulus)
    if args.smoke:
        first_by_category = {
            category: next(opener for opener in openers if opener.category == category)
            for category in ("mirror", "dominance", "neutral")
        }
        openers = tuple(first_by_category.values())
        args.max_new_tokens = min(args.max_new_tokens, 16)
    base_model = load_adapter_base(model_source, dtype=torch.bfloat16, device=device)
    if args.smoke:
        base_model, smoke_adapters = _build_smoke_adapters(
            base_model,
            source_adapters=(args.self_adapter, args.group_adapter),
            output_root=args.output / "smoke-adapters",
            model_id=args.model,
        )
        args.self_adapter, args.group_adapter = smoke_adapters
        args.adapter_digests = {
            "self": adapter_digest(args.self_adapter),
            "group": adapter_digest(args.group_adapter),
        }
    _assert_adapter_shapes_match_model(args.self_adapter, base_model)
    _assert_adapter_shapes_match_model(args.group_adapter, base_model)
    hidden_size = _hidden_size(base_model)
    batch_size = derive_batch_size(
        device,
        max_sequence_length=8192,
        hidden_size=hidden_size,
        cap=args.batch_cap,
    )
    logger.info("argument-prior device=%s derived_batch_size=%d", device, batch_size)
    attached = attach_adapter(base_model, args.self_adapter, args.model)
    assert_one_adapter_config((args.self_adapter, args.group_adapter))
    opener_path = args.output / "opener_records.jsonl"
    completed = load_completed_keys(opener_path)
    written = 0
    resumed = 0
    by_condition = {"base": True, "self": False, "group": False}
    for condition in MODEL_CONDITIONS:
        if condition == "group":
            attached = attach_adapter(
                base_model, args.group_adapter, args.model, existing=attached.peft_model
            )
        for row, opener, total, mean, token_count in _score_openers(
            cast("Any", attached.peft_model),
            cast("Any", tokenizer),
            rows,
            openers,
            batch_size=batch_size,
            disable_adapter=by_condition[condition],
        ):
            key = content_key(
                "opener",
                stimulus_digest,
                args.model,
                str(args.model_path or ""),
                condition,
                row.game_id,
                row.framing_id,
                row.prompt_id,
                row.prompt,
                row.label_print_order,
                opener.identifier,
                opener.text,
                args.adapter_digests.get(condition, ""),
            )
            if key in completed:
                resumed += 1
                continue
            append_jsonl(
                opener_path,
                {
                    "key": key,
                    "record": "opener_logprob",
                    "model_condition": condition,
                    "game_id": row.game_id,
                    "framing_id": row.framing_id,
                    "prompt_id": row.prompt_id,
                    "opener_id": opener.identifier,
                    "opener_category": opener.category,
                    "adapter_available": condition != "base",
                    "logprob_sum": total,
                    "logprob_mean": mean,
                    "token_count": token_count,
                    "device": str(device),
                    "batch_size": batch_size,
                },
            )
            completed.add(key)
            written += 1
    all_records = []
    with opener_path.open(encoding="utf-8") as handle:
        all_records = [json.loads(line) for line in handle if line.strip()]
    summary = _summarize_openers(all_records)
    summary["behavior"] = behavior_summary(args.records_root, strict=not args.smoke)
    summary["run"] = {
        "model": args.model,
        "model_path": str(args.model_path or ""),
        "stimulus_sha256": stimulus_digest,
        "device": memory,
        "derived_batch_size": batch_size,
        "written": written,
        "resumed": resumed,
        "adapter_available": True,
        "smoke_base_reused": bool(args.smoke),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    donor_base: PreTrainedModel | None = base_model
    donor_attached: Any | None = attached
    if device.type == "cuda":
        donor_base = None
        donor_attached = None
        del attached
        del base_model
        torch.cuda.empty_cache()
    donor_summary = _run_donor_part(
        args,
        donor_base,
        tokenizer,
        rows=rows,
        attached=donor_attached,
        output_path=args.output / "donor_continuations.jsonl",
        stimulus_digest=stimulus_digest,
        device=device,
    )
    summary["donor"] = donor_summary
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table_lines = [
        (
            "model_condition\tgame_id\tframing_id\tmirror_minus_dominance\t"
            "mirror_minus_neutral\tdominance_minus_neutral"
        )
    ]
    table_lines.extend(
        "\t".join(
            str(row.get(field))
            for field in (
                "model_condition",
                "game_id",
                "framing_id",
                "mirror_minus_dominance",
                "mirror_minus_neutral",
                "dominance_minus_neutral",
            )
        )
        for row in summary["contexts"]
    )
    (args.output / "summary.tsv").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    readout_lines = [
        "# Argument-prior map",
        "",
        f"Model: `{args.model}`",
        f"Opener records written: {written}; resumed: {resumed}.",
        (
            f"Donors: {donor_summary['donors']}; donor records written: "
            f"{donor_summary['written']}; resumed: {donor_summary['resumed']}."
        ),
        "",
        (
            "Donor cooperation rates are in `summary.json` under `donor.cooperation_rates`; "
            "raw continuations remain in `donor_continuations.jsonl`."
        ),
    ]
    (args.output / "readout.md").write_text("\n".join(readout_lines) + "\n", encoding="utf-8")
    return summary


def _summarize_openers(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate mirror/dominance/neutral opener contrasts by model and context."""
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[
            (
                str(record["model_condition"]),
                str(record["game_id"]),
                str(record["framing_id"]),
            )
        ][str(record["opener_category"])].append(float(record["logprob_mean"]))
    rows: list[dict[str, Any]] = []
    for (condition, game_id, framing_id), categories in sorted(grouped.items()):
        means = {
            category: sum(values) / len(values) for category, values in categories.items() if values
        }
        neutral = means.get("neutral")
        mirror = means.get("mirror")
        dominance = means.get("dominance")
        rows.append(
            {
                "model_condition": condition,
                "game_id": game_id,
                "framing_id": framing_id,
                "category_means": means,
                "mirror_minus_dominance": (
                    mirror - dominance if mirror is not None and dominance is not None else None
                ),
                "mirror_minus_neutral": (
                    mirror - neutral if mirror is not None and neutral is not None else None
                ),
                "dominance_minus_neutral": (
                    dominance - neutral if dominance is not None and neutral is not None else None
                ),
            }
        )
    by_context: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = row["mirror_minus_dominance"]
        if value is not None:
            by_context[(str(row["game_id"]), str(row["framing_id"]), "shift")][
                str(row["model_condition"])
            ] = float(value)
    shifts = []
    for (game_id, framing_id, _), values in sorted(by_context.items()):
        base = values.get("base")
        shifts.append(
            {
                "game_id": game_id,
                "framing_id": framing_id,
                "adapter_shifts": {
                    condition: value - base
                    for condition, value in values.items()
                    if condition != "base" and base is not None
                },
            }
        )
    return {"contexts": rows, "adapter_shifts": shifts}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument-prior CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="local snapshot to load while retaining --model as the adapter/base identity",
    )
    parser.add_argument("--self-adapter", type=Path, default=SELF_ADAPTER)
    parser.add_argument("--group-adapter", type=Path, default=GROUP_ADAPTER)
    parser.add_argument("--stimulus", type=Path, default=DEFAULT_STIMULUS)
    parser.add_argument("--records-root", type=Path, default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rows-per-cell", type=int, default=4)
    parser.add_argument("--batch-cap", type=int, default=32)
    parser.add_argument("--donors", type=int, default=4)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    """Run the argument-prior map CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_argument_prior(build_parser().parse_args())


if __name__ == "__main__":
    main()
