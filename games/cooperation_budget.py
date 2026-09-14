"""CPU-only count, disk, and measured-time planning for cooperation generalisation.

The planner consumes runtime manifests and JSONL artifacts.  It does not resolve a model id,
construct a model, touch CUDA, or replace a measured value with a remembered ETA.  Callers that
already obtained a parameter count from :func:`games.sizing.count_meta_parameters` can pass it in
``ModelMetadata``; tests and print-plan code should pass metadata directly.

Every disk number in this module is an estimate until a smoke writes the corresponding artifact.
The measured-budget aggregator is deliberately stricter: it requires positive elapsed times for
every incurred GPU phase, positive probe-derived prices for every future phase, and a persisted
measured step cap before it reserves a training block longer than the initial limit.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from types import ModuleType

BYTES_PER_GIB = 1024**3
BF16_BYTES = 2
FP32_BYTES = 4
DEFAULT_LORA_RANK = 16
DEFAULT_CHECKPOINT_COUNT = 21
DEFAULT_CAPTURE_STATES = 2
DEFAULT_CAPTURE_POOLINGS = 2
DEFAULT_CAPTURE_BOUNDARIES = 3
DEFAULT_LENS_STATES = 2
DEFAULT_CAPTURE_DTYPE_BYTES = FP32_BYTES
DEFAULT_LENS_DTYPE_BYTES = BF16_BYTES
DEFAULT_ACCUMULATOR_DTYPE_BYTES = FP32_BYTES
DEFAULT_HEADROOM_FRACTION = 0.20
INITIAL_MAX_STEPS = 20
SHAPE_DIMENSIONS = 2

# These names are the contract of the persisted budget artifact.  Prelaunch values are complete
# phase-receipt elapsed times.  The endpoint values reserve work that has not run yet, so the
# already-incurred baseline appears only in ``prelaunch`` and ``final_eval`` appears only here.
REQUIRED_THROUGHPUT_TIMINGS: tuple[str, ...] = ("generation", "backward", "optimizer")
REQUIRED_PRELAUNCH_TIMINGS: tuple[str, ...] = (
    "smoke_0_6b",
    "tiny_9b_update",
    "tiny_9b_serving",
    "tiny_9b_capture",
    "tiny_9b_gradient",
    "matching_sampler_screen",
    "throughput_probe",
    "baseline",
)
REQUIRED_ENDPOINT_TIMINGS: tuple[str, ...] = (
    "final_eval",
    "capture",
    "lens",
    "intervention",
)
REQUIRED_MEASUREMENTS: tuple[str, ...] = (
    *REQUIRED_THROUGHPUT_TIMINGS,
    *REQUIRED_ENDPOINT_TIMINGS,
    "headroom",
)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _positive_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a finite positive number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return number


def _dtype_bytes(value: object, *, name: str) -> int:
    if isinstance(value, str):
        normalized = value.lower().replace("torch.", "")
        aliases = {
            "bf16": BF16_BYTES,
            "bfloat16": BF16_BYTES,
            "float16": 2,
            "fp16": 2,
            "fp32": FP32_BYTES,
            "float32": FP32_BYTES,
        }
        if normalized not in aliases:
            raise ValueError(f"{name} has unsupported dtype {value!r}")
        return aliases[normalized]
    return _positive_int(value, name=name)


@dataclass(frozen=True, slots=True)
class LoraTarget:
    """One discovered linear target family and its meta-device shape."""

    name: str
    count: int
    in_features: int
    out_features: int

    def __post_init__(self) -> None:
        """Reject an incomplete or non-positive discovered target shape."""
        _positive_int(self.count, name=f"LoRA target {self.name} count")
        _positive_int(self.in_features, name=f"LoRA target {self.name} in_features")
        _positive_int(self.out_features, name=f"LoRA target {self.name} out_features")

    def parameter_count(self, rank: int) -> int:
        """Return the adapter parameter count for this target family."""
        rank = _positive_int(rank, name="LoRA rank")
        return self.count * rank * (self.in_features + self.out_features)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Architecture metadata supplied by a config/meta inspection, without model weights."""

    parameter_count: int
    hidden_size: int
    n_layers: int
    dtype_bytes: int = BF16_BYTES
    source_layers: int | None = None
    lora_rank: int = DEFAULT_LORA_RANK
    lora_parameter_count: int | None = None
    lora_targets: tuple[LoraTarget, ...] = ()

    def __post_init__(self) -> None:
        """Reject metadata that could make the estimate look smaller than it is."""
        _positive_int(self.parameter_count, name="parameter_count")
        _positive_int(self.hidden_size, name="hidden_size")
        _positive_int(self.n_layers, name="n_layers")
        _positive_int(self.dtype_bytes, name="dtype_bytes")
        _positive_int(self.lora_rank, name="lora_rank")
        if self.source_layers is not None:
            _positive_int(self.source_layers, name="source_layers")
        if self.lora_parameter_count is not None:
            _positive_int(self.lora_parameter_count, name="lora_parameter_count")

    @property
    def lens_layers(self) -> int:
        """Return the source-layer count used by the Jacobian fit arithmetic."""
        return self.source_layers if self.source_layers is not None else self.n_layers

    @property
    def derived_lora_parameter_count(self) -> int | None:
        """Return explicit or shape-derived trainable parameters when available."""
        if self.lora_parameter_count is not None:
            return self.lora_parameter_count
        if not self.lora_targets:
            return None
        return sum(target.parameter_count(self.lora_rank) for target in self.lora_targets)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ModelMetadata:
        """Build metadata from config fields or a saved LoRA target inspection.

        ``parameter_count`` may be produced by ``games.sizing.count_meta_parameters``.  The
        optional ``lora_targets`` shape list is the output of a separate meta-module inspection;
        a target-name count without dimensions is retained as unknown instead of guessed.
        """
        config_payload = payload.get("text_config")
        config = config_payload if isinstance(config_payload, Mapping) else payload
        parameter_count = payload.get("parameter_count", payload.get("num_parameters"))
        hidden_size = config.get("hidden_size", config.get("hidden_dim"))
        n_layers = config.get("n_layers", config.get("num_hidden_layers", config.get("num_layers")))
        if parameter_count is None or hidden_size is None or n_layers is None:
            raise ValueError(
                "model metadata needs parameter_count, hidden_size, and n_layers "
                f"(got keys {sorted(payload)})"
            )
        raw_targets = payload.get("lora_targets", payload.get("lora_target_shapes", ()))
        targets = tuple(_parse_lora_targets(raw_targets))
        explicit_lora = payload.get(
            "lora_parameter_count",
            payload.get("lora_target_parameter_count", payload.get("trainable_parameters")),
        )
        return cls(
            parameter_count=_positive_int(parameter_count, name="parameter_count"),
            hidden_size=_positive_int(hidden_size, name="hidden_size"),
            n_layers=_positive_int(n_layers, name="n_layers"),
            dtype_bytes=_dtype_bytes(
                payload.get("dtype_bytes", payload.get("dtype", config.get("dtype", BF16_BYTES))),
                name="dtype",
            ),
            source_layers=(
                None
                if payload.get("source_layers", payload.get("lens_layers")) is None
                else _positive_int(
                    payload.get("source_layers", payload.get("lens_layers")), name="source_layers"
                )
            ),
            lora_rank=_positive_int(payload.get("lora_rank", DEFAULT_LORA_RANK), name="lora_rank"),
            lora_parameter_count=(
                None
                if explicit_lora is None
                else _positive_int(explicit_lora, name="lora_parameter_count")
            ),
            lora_targets=targets,
        )


def _parse_lora_targets(raw: object) -> Iterable[LoraTarget]:  # noqa: C901, PLR0912
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        # ``discover_lora_targets`` returns module_counts.  It does not carry dimensions, so only
        # a nested target_shapes mapping is sufficient to calculate adapter bytes.
        if "target_shapes" in raw:
            raw = raw["target_shapes"]
        elif "module_counts" in raw:
            # Counts from ``discover_lora_targets`` identify families but do not identify their
            # in/out dimensions.  Keep the adapter estimate unknown rather than inventing a shape.
            return ()
    if isinstance(raw, Mapping):
        result: list[LoraTarget] = []
        for name, shape in raw.items():
            if (
                isinstance(shape, Sequence)
                and not isinstance(shape, str)
                and len(shape) == SHAPE_DIMENSIONS
            ):
                result.append(
                    LoraTarget(
                        str(name),
                        1,
                        _positive_int(shape[0], name="in_features"),
                        _positive_int(shape[1], name="out_features"),
                    )
                )
            elif isinstance(shape, Mapping):
                result.append(
                    LoraTarget(
                        name=str(shape.get("name", name)),
                        count=_positive_int(shape.get("count", 1), name="lora target count"),
                        in_features=_positive_int(shape.get("in_features"), name="in_features"),
                        out_features=_positive_int(shape.get("out_features"), name="out_features"),
                    )
                )
            elif isinstance(shape, int):
                return ()
            else:
                raise ValueError("lora target mappings need [in_features, out_features] shapes")
        return result
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise TypeError("lora_targets must be a list or mapping")
    result = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise TypeError(f"lora_targets row {index} must be an object")
        result.append(
            LoraTarget(
                name=str(value.get("name", f"target-{index}")),
                count=_positive_int(value.get("count", 1), name="lora target count"),
                in_features=_positive_int(value.get("in_features"), name="in_features"),
                out_features=_positive_int(value.get("out_features"), name="out_features"),
            )
        )
    return result


@dataclass(frozen=True, slots=True)
class TrainingInputCounts:
    """Counts from the text-free training manifest and its rendered corpus."""

    manifest_rows: int
    corpus_rows: int

    @property
    def rows(self) -> int:
        """Return the corpus row count used by the sampler."""
        return self.corpus_rows


@dataclass(frozen=True, slots=True)
class BehaviorInputCounts:
    """Expanded behavior requests reported by the cooperation evaluator."""

    pairs: int
    rendered_prompts: int
    completions: int
    by_family: Mapping[str, int]
    context_responses: int = 0
    allocation_responses: int = 0


@dataclass(frozen=True, slots=True)
class SurveyInputCounts:
    """Runtime response counts for the required survey and decision-theory cells."""

    core_responses: int
    prosocialness_responses: int
    decision_theory_responses: int

    @property
    def responses_per_checkpoint(self) -> int:
        """Return all required non-behavior responses at one checkpoint."""
        return self.core_responses + self.prosocialness_responses + self.decision_theory_responses


@dataclass(frozen=True, slots=True)
class GenerationBudget:
    """Completion and token counts across training, endpoints, and intervention."""

    training_completions_per_step: int
    training_completions: int
    training_tokens: int
    behavior_responses_per_checkpoint: int
    context_responses_per_checkpoint: int
    allocation_responses_per_checkpoint: int
    survey_responses_per_checkpoint: int
    endpoint_responses: int
    intervention_responses: int
    total_completions: int
    total_generated_tokens: int


@dataclass(frozen=True, slots=True)
class DiskBudget:
    """Estimated retained bytes, with unknown terms kept visible."""

    base_model_bf16_bytes: int
    lora_parameter_count: int | None
    lora_checkpoint_bytes_each: int | None
    checkpoint_count: int
    checkpoint_bytes_total: int | None
    construct_rows: int
    capture_states: int
    capture_poolings: int
    capture_boundaries: int
    capture_bytes_each: int
    capture_bytes_total: int
    lens_stimulus_rows: int
    lens_states: int
    lens_matrix_bytes_each: int
    accumulator_bytes_each: int
    lens_bytes_each: int
    lens_bytes_total: int
    estimated_bytes: int
    unknown_estimates: tuple[str, ...]

    @property
    def estimated_gib(self) -> float:
        """Return the known estimate in binary GiB."""
        return self.estimated_bytes / BYTES_PER_GIB


@dataclass(frozen=True, slots=True)
class BudgetPlan:
    """Complete pre-GPU count and disk plan."""

    training: TrainingInputCounts
    behavior: BehaviorInputCounts
    survey: SurveyInputCounts
    generation: GenerationBudget
    disk: DiskBudget
    max_steps: int
    measured_replacements_required: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for print-plan output."""
        return cast("dict[str, object]", asdict(self))


def _json_records(path: Path) -> list[Mapping[str, object]]:  # noqa: C901
    if not path.is_file():
        raise FileNotFoundError(f"runtime budget input does not exist: {path}")
    if path.suffix == ".jsonl":
        records: list[Mapping[str, object]] = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"{path} row {index} is not a JSON object")
            records.append(cast("Mapping[str, object]", value))
        if not records:
            raise ValueError(f"{path} contains no JSONL records")
        return records
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = payload.get("rows", payload.get("pairs"))
    else:
        rows = None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON list or rows/pairs list")
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError(f"{path} contains a non-object row")
    return [cast("Mapping[str, object]", row) for row in rows]


def count_jsonl_records(path: Path) -> int:
    """Count non-empty JSONL objects, rejecting malformed or empty runtime artifacts."""
    return len(_json_records(path))


def _manifest_row_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("row_count"), int):
        return _positive_int(payload["row_count"], name="training manifest row_count")
    return len(_json_records(path))


def _manifest_corpus_path(manifest: Path) -> Path | None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("corpus_path"), str):
        return None
    path = Path(payload["corpus_path"])
    return path if path.is_absolute() else manifest.parent / path


def count_training_inputs(manifest: Path, corpus: Path | None = None) -> TrainingInputCounts:
    """Count the frozen training manifest and rendered corpus, requiring agreement."""
    if not manifest.is_file():
        raise FileNotFoundError(f"training manifest does not exist: {manifest}")
    manifest_rows = _manifest_row_count(manifest)
    resolved_corpus = corpus or _manifest_corpus_path(manifest)
    if resolved_corpus is None:
        raise ValueError(f"{manifest} does not identify a training corpus")
    corpus_rows = count_jsonl_records(resolved_corpus)
    if manifest_rows != corpus_rows:
        raise ValueError(
            f"training manifest row_count={manifest_rows} disagrees with corpus rows={corpus_rows}"
        )
    return TrainingInputCounts(manifest_rows=manifest_rows, corpus_rows=corpus_rows)


def count_behavior_inputs(path: Path) -> BehaviorInputCounts:
    """Use the cooperation evaluator's expansion counts, retaining family accounting."""
    evaluator = _cooperation_evals()
    manifest = evaluator.load_behavior_manifest(path)
    roster = evaluator.expand_behavior_roster(manifest)
    counts = roster.counts
    evaluator_plan = evaluator.print_plan(path)
    context_counts = evaluator_plan.get("exact_context")
    allocation_counts = evaluator_plan.get("allocation")
    if not isinstance(context_counts, Mapping) or not isinstance(allocation_counts, Mapping):
        raise TypeError(
            "cooperation evaluator print_plan omitted exact-context or allocation counts"
        )
    context_responses = _positive_int(
        _positive_int(context_counts.get("forecast_requests"), name="forecast requests")
        + _positive_int(context_counts.get("normative_requests"), name="normative requests"),
        name="exact-context responses",
    )
    allocation_responses = _positive_int(
        allocation_counts.get("responses"), name="allocation responses"
    )
    return BehaviorInputCounts(
        pairs=counts.n_pairs,
        rendered_prompts=counts.n_rendered_prompts,
        completions=counts.n_completions,
        by_family=dict(counts.by_family),
        context_responses=context_responses,
        allocation_responses=allocation_responses,
    )


def count_survey_inputs(
    data_dir: Path,
    *,
    survey_samples: int = 2,
    multiple_choice_samples: int = 2,
    open_ended_samples: int = 2,
) -> SurveyInputCounts:
    """Recompute core, prosocialness, and local DT responses from runtime survey files."""
    _positive_int(survey_samples, name="survey_samples")
    evaluator = _cooperation_evals()
    core = evaluator.survey_leg_counts("core", data_dir=data_dir)
    prosocialness = evaluator.survey_leg_counts("prosocialness", data_dir=data_dir)
    del survey_samples  # ``survey_leg_counts`` carries the runtime membership and multiplicity.
    dt = evaluator.local_dt_counts(
        multiple_choice_samples=multiple_choice_samples,
        open_ended_samples=open_ended_samples,
    )
    return SurveyInputCounts(
        core_responses=_positive_int(core.n_responses, name="core survey responses"),
        prosocialness_responses=_positive_int(
            prosocialness.n_responses, name="prosocialness survey responses"
        ),
        decision_theory_responses=_positive_int(dt["n_responses"], name="DT responses"),
    )


def _paths(value: Path | Sequence[Path] | Mapping[str, Path]) -> tuple[Path, ...]:
    if isinstance(value, Path):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(value.values())
    return tuple(value)


def _count_paths(value: Path | Sequence[Path] | Mapping[str, Path]) -> int:
    paths = _paths(value)
    if not paths:
        raise ValueError("runtime JSONL input list cannot be empty")
    return sum(count_jsonl_records(path) for path in paths)


def _cooperation_evals() -> ModuleType:
    """Load evaluator counting code only when a runtime manifest is actually counted."""
    return importlib.import_module("games.cooperation_evals")


def estimate_disk_budget(  # noqa: PLR0913
    *,
    model: ModelMetadata,
    max_steps: int,
    construct_rows: int,
    lens_stimulus_rows: int,
    capture_states: int = DEFAULT_CAPTURE_STATES,
    capture_poolings: int = DEFAULT_CAPTURE_POOLINGS,
    capture_boundaries: int = DEFAULT_CAPTURE_BOUNDARIES,
    capture_layers: int | None = None,
    capture_hidden_size: int | None = None,
    capture_dtype_bytes: int = DEFAULT_CAPTURE_DTYPE_BYTES,
    lens_states: int = DEFAULT_LENS_STATES,
    lens_dtype_bytes: int = DEFAULT_LENS_DTYPE_BYTES,
    accumulator_dtype_bytes: int = DEFAULT_ACCUMULATOR_DTYPE_BYTES,
    checkpoint_count: int | None = None,
) -> DiskBudget:
    """Estimate deduplicated base, checkpoints, pooled captures, and Jacobian artifacts."""
    max_steps = _positive_int(max_steps, name="max_steps")
    construct_rows = _positive_int(construct_rows, name="construct_rows")
    lens_stimulus_rows = _positive_int(lens_stimulus_rows, name="lens_stimulus_rows")
    capture_states = _positive_int(capture_states, name="capture_states")
    capture_poolings = _positive_int(capture_poolings, name="capture_poolings")
    capture_boundaries = _positive_int(capture_boundaries, name="capture_boundaries")
    lens_states = _positive_int(lens_states, name="lens_states")
    layers = _positive_int(
        model.n_layers if capture_layers is None else capture_layers, name="capture_layers"
    )
    hidden = _positive_int(
        model.hidden_size if capture_hidden_size is None else capture_hidden_size,
        name="capture_hidden_size",
    )
    capture_dtype_bytes = _dtype_bytes(capture_dtype_bytes, name="capture_dtype_bytes")
    lens_dtype_bytes = _dtype_bytes(lens_dtype_bytes, name="lens_dtype_bytes")
    accumulator_dtype_bytes = _dtype_bytes(accumulator_dtype_bytes, name="accumulator_dtype_bytes")
    checkpoints = _positive_int(
        checkpoint_count if checkpoint_count is not None else max_steps + 1,
        name="checkpoint_count",
    )

    lora_parameter_count = model.derived_lora_parameter_count
    lora_checkpoint_bytes_each = None
    if lora_parameter_count is not None:
        adapter_bytes = lora_parameter_count * model.dtype_bytes
        # Adam keeps two fp32 moments.  Small JSON/RNG/scheduler files are deliberately excluded;
        # the smoke's measured replacement must cover them before the research block.
        lora_checkpoint_bytes_each = adapter_bytes + lora_parameter_count * 2 * FP32_BYTES
    checkpoint_bytes_total = (
        None if lora_checkpoint_bytes_each is None else checkpoints * lora_checkpoint_bytes_each
    )
    capture_bytes_each = layers * hidden * capture_dtype_bytes
    capture_bytes_total = (
        construct_rows * capture_states * capture_poolings * capture_boundaries * capture_bytes_each
    )
    lens_matrix_bytes_each = model.lens_layers * model.hidden_size**2 * lens_dtype_bytes
    accumulator_bytes_each = model.lens_layers * model.hidden_size**2 * accumulator_dtype_bytes
    lens_bytes_each = lens_matrix_bytes_each + accumulator_bytes_each
    lens_bytes_total = lens_states * lens_bytes_each
    unknown: list[str] = []
    unknown.append("base model bytes on disk")
    if lora_checkpoint_bytes_each is None:
        unknown.append("LoRA checkpoint bytes (no target shapes or explicit trainable count)")
    unknown.extend(("rollout text bytes", "checkpoint metadata bytes"))
    known = model.parameter_count * BF16_BYTES + capture_bytes_total + lens_bytes_total
    if checkpoint_bytes_total is not None:
        known += checkpoint_bytes_total
    return DiskBudget(
        base_model_bf16_bytes=model.parameter_count * BF16_BYTES,
        lora_parameter_count=lora_parameter_count,
        lora_checkpoint_bytes_each=lora_checkpoint_bytes_each,
        checkpoint_count=checkpoints,
        checkpoint_bytes_total=checkpoint_bytes_total,
        construct_rows=construct_rows,
        capture_states=capture_states,
        capture_poolings=capture_poolings,
        capture_boundaries=capture_boundaries,
        capture_bytes_each=capture_bytes_each,
        capture_bytes_total=capture_bytes_total,
        lens_stimulus_rows=lens_stimulus_rows,
        lens_states=lens_states,
        lens_matrix_bytes_each=lens_matrix_bytes_each,
        accumulator_bytes_each=accumulator_bytes_each,
        lens_bytes_each=lens_bytes_each,
        lens_bytes_total=lens_bytes_total,
        estimated_bytes=known,
        unknown_estimates=tuple(unknown),
    )


def plan_budget(  # noqa: PLR0913
    *,
    training_manifest: Path,
    training_corpus: Path | None,
    behavior_manifest: Path,
    survey_data_dir: Path,
    construct_stimuli: Path | Sequence[Path] | Mapping[str, Path],
    lens_stimuli: Path | Sequence[Path] | Mapping[str, Path],
    model: ModelMetadata | Mapping[str, object],
    max_steps: int = INITIAL_MAX_STEPS,
    prompts_per_step: int = 8,
    group_size: int = 8,
    oversample: int = 1,
    completion_tokens: int = 32_768,
    capture_states: int = DEFAULT_CAPTURE_STATES,
    capture_poolings: int = DEFAULT_CAPTURE_POOLINGS,
    capture_boundaries: int = DEFAULT_CAPTURE_BOUNDARIES,
    capture_layers: int | None = None,
    capture_hidden_size: int | None = None,
    capture_dtype_bytes: int = DEFAULT_CAPTURE_DTYPE_BYTES,
    lens_states: int = DEFAULT_LENS_STATES,
    lens_dtype_bytes: int = DEFAULT_LENS_DTYPE_BYTES,
    accumulator_dtype_bytes: int = DEFAULT_ACCUMULATOR_DTYPE_BYTES,
    checkpoint_count: int | None = None,
    intervention_prompts: int = 8,
    intervention_conditions: int = 3,
    intervention_samples: int = 2,
) -> BudgetPlan:
    """Build the pre-GPU plan from the exact runtime files and injected architecture metadata."""
    max_steps = _positive_int(max_steps, name="max_steps")
    prompts_per_step = _positive_int(prompts_per_step, name="prompts_per_step")
    group_size = _positive_int(group_size, name="group_size")
    oversample = _positive_int(oversample, name="oversample")
    completion_tokens = _positive_int(completion_tokens, name="completion_tokens")
    intervention_prompts = _positive_int(intervention_prompts, name="intervention_prompts")
    intervention_conditions = _positive_int(intervention_conditions, name="intervention_conditions")
    intervention_samples = _positive_int(intervention_samples, name="intervention_samples")
    resolved_model = (
        model if isinstance(model, ModelMetadata) else ModelMetadata.from_mapping(model)
    )
    training = count_training_inputs(training_manifest, training_corpus)
    behavior = count_behavior_inputs(behavior_manifest)
    survey = count_survey_inputs(survey_data_dir)
    construct_rows = _count_paths(construct_stimuli)
    lens_rows = _count_paths(lens_stimuli)
    training_per_step = prompts_per_step * group_size * oversample
    training_completions = max_steps * training_per_step
    behavior_per_checkpoint = behavior.completions
    context_per_checkpoint = behavior.context_responses
    allocation_per_checkpoint = behavior.allocation_responses
    survey_per_checkpoint = survey.responses_per_checkpoint
    endpoint_responses = 2 * (
        behavior_per_checkpoint
        + context_per_checkpoint
        + allocation_per_checkpoint
        + survey_per_checkpoint
    )
    intervention_responses = intervention_prompts * intervention_conditions * intervention_samples
    total_completions = training_completions + endpoint_responses + intervention_responses
    generation = GenerationBudget(
        training_completions_per_step=training_per_step,
        training_completions=training_completions,
        training_tokens=training_completions * completion_tokens,
        behavior_responses_per_checkpoint=behavior_per_checkpoint,
        context_responses_per_checkpoint=context_per_checkpoint,
        allocation_responses_per_checkpoint=allocation_per_checkpoint,
        survey_responses_per_checkpoint=survey_per_checkpoint,
        endpoint_responses=endpoint_responses,
        intervention_responses=intervention_responses,
        total_completions=total_completions,
        total_generated_tokens=total_completions * completion_tokens,
    )
    disk = estimate_disk_budget(
        model=resolved_model,
        max_steps=max_steps,
        construct_rows=construct_rows,
        lens_stimulus_rows=lens_rows,
        capture_states=capture_states,
        capture_poolings=capture_poolings,
        capture_boundaries=capture_boundaries,
        capture_layers=capture_layers,
        capture_hidden_size=capture_hidden_size,
        capture_dtype_bytes=capture_dtype_bytes,
        lens_states=lens_states,
        lens_dtype_bytes=lens_dtype_bytes,
        accumulator_dtype_bytes=accumulator_dtype_bytes,
        checkpoint_count=checkpoint_count,
    )
    measured_replacements = (
        "checkpoint bytes per saved step",
        "rollout text bytes per completion",
        "pooled capture bytes per state, pooling, and measurement boundary",
        "Jacobian lens and accumulator bytes",
        "generation, backward, optimizer, endpoint, capture, lens, intervention timings",
    )
    return BudgetPlan(
        training=training,
        behavior=behavior,
        survey=survey,
        generation=generation,
        disk=disk,
        max_steps=max_steps,
        measured_replacements_required=measured_replacements,
    )


def _load_json(source: Path | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(source, Path):
        if not source.is_file():
            raise FileNotFoundError(f"measurement artifact does not exist: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        payload = source
    if not isinstance(payload, Mapping):
        raise TypeError("measurement artifact must be a JSON object")
    return cast("Mapping[str, object]", payload)


def _walk_values(value: object, *, key_names: set[str]) -> list[object]:
    found: list[object] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("/", "_").replace("-", "_")
            if normalized in key_names:
                found.append(child)
            found.extend(_walk_values(child, key_names=key_names))
    elif isinstance(value, Sequence) and not isinstance(value, str):
        for child in value:
            found.extend(_walk_values(child, key_names=key_names))
    return found


def _timing_from_payload(payload: Mapping[str, object], name: str) -> float:
    aliases = {
        "generation": {
            "generation",
            "generation_seconds",
            "generation_seconds_measured",
            "median_generation_seconds",
            "generate_s",
            "generate_seconds",
            "median_generate_seconds",
            "timing_generate_s",
        },
        "backward": {
            "backward",
            "backward_seconds",
            "backward_seconds_measured",
            "median_backward_seconds",
            "backward_s",
            "timing_backward_s",
        },
        "optimizer": {
            "optimizer",
            "optimizer_seconds",
            "optimizer_step_seconds",
            "optimizer_step_seconds_measured",
            "median_optimizer_seconds",
            "median_optimizer_step_seconds",
            "optimizer_step_s",
            "timing_optimizer_step_s",
        },
    }
    values = _walk_values(payload, key_names=aliases[name])
    if not values:
        raise ValueError(f"persisted throughput JSON is missing positive {name} timing")
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, str):
            if not value:
                raise ValueError(f"persisted throughput JSON has empty {name} timing list")
            numbers = [_positive_number(item, name=f"{name} timing") for item in value]
            return float(statistics.median(numbers))
        return _positive_number(value, name=f"{name} timing")
    raise ValueError(f"persisted throughput JSON is missing positive {name} timing")


def _endpoint_timings(payload: Mapping[str, object]) -> dict[str, float]:
    raw: object = payload.get("endpoints", payload.get("endpoint_timings"))
    if raw is None and any(name in payload for name in REQUIRED_ENDPOINT_TIMINGS):
        raw = payload
    if not isinstance(raw, Mapping):
        raise TypeError("measurement artifact endpoint timings must be a JSON object")
    values = cast("Mapping[str, object]", raw)
    result: dict[str, float] = {}
    if "final_eval" not in values:
        raise ValueError("measurement artifact is missing positive final_eval timing")
    result["final_eval"] = _positive_number(values["final_eval"], name="final_eval timing")
    for name in ("capture", "lens", "intervention"):
        if name not in values:
            raise ValueError(f"measurement artifact is missing positive {name} endpoint timing")
        result[name] = _positive_number(values[name], name=f"{name} endpoint timing")
    return result


def _prelaunch_timings(payload: Mapping[str, object]) -> dict[str, float]:
    """Read complete elapsed times for every GPU phase incurred before the budget gate."""
    raw = payload.get("prelaunch")
    if not isinstance(raw, Mapping):
        raise TypeError("measurement artifact prelaunch timings must be a JSON object")
    values = cast("Mapping[str, object]", raw)
    result: dict[str, float] = {}
    for name in REQUIRED_PRELAUNCH_TIMINGS:
        if name not in values:
            raise ValueError(f"measurement artifact is missing positive {name} prelaunch timing")
        result[name] = _positive_number(values[name], name=f"{name} prelaunch timing")
    return result


def _headroom(payload: Mapping[str, object]) -> tuple[float, float]:
    fraction = payload.get("headroom_fraction")
    if fraction is not None:
        fraction_value = _positive_number(fraction, name="headroom_fraction")
        if fraction_value >= 1:
            raise ValueError(f"headroom_fraction must be below one, got {fraction_value}")
        return fraction_value, 0.0
    seconds = payload.get("headroom_seconds", payload.get("headroom"))
    if seconds is None:
        raise ValueError("measurement artifact is missing positive persisted headroom")
    return 0.0, _positive_number(seconds, name="headroom_seconds")


def _prelaunch_contract(payload: Mapping[str, object]) -> tuple[int, int, int, int]:
    """Read the persisted token plan and measured disk evidence for this launch."""
    completion_token_cap = _positive_int(
        payload.get("completion_token_cap"), name="completion_token_cap"
    )
    planned_generation_completions = _positive_int(
        payload.get("planned_generation_completions"), name="planned_generation_completions"
    )
    planned_generated_token_cap = _positive_int(
        payload.get("planned_generated_token_cap"), name="planned_generated_token_cap"
    )
    expected_token_cap = completion_token_cap * planned_generation_completions
    if planned_generated_token_cap != expected_token_cap:
        raise ValueError(
            "planned_generated_token_cap must equal completion_token_cap * "
            f"planned_generation_completions ({expected_token_cap}), got "
            f"{planned_generated_token_cap}"
        )

    candidates: list[object] = [
        payload[key]
        for key in ("disk_actual_bytes", "actual_disk_bytes", "measured_disk_bytes", "disk_bytes")
        if key in payload
    ]
    disk_payload = payload.get("disk")
    if isinstance(disk_payload, Mapping):
        candidates.extend(
            disk_payload[key]
            for key in ("actual_bytes", "disk_actual_bytes", "measured_bytes")
            if key in disk_payload
        )
    if not candidates:
        raise ValueError("measurement artifact is missing positive disk actual bytes")
    disk_actual_bytes = _positive_int(candidates[0], name="disk_actual_bytes")
    if any(
        _positive_int(value, name="disk_actual_bytes") != disk_actual_bytes
        for value in candidates[1:]
    ):
        raise ValueError("measurement artifact contains mismatched disk actual bytes")
    return (
        completion_token_cap,
        planned_generation_completions,
        planned_generated_token_cap,
        disk_actual_bytes,
    )


def _measured_cap(payload: Mapping[str, object]) -> int | None:
    value = payload.get("measured_max_steps", payload.get("max_steps_cap"))
    if value is None:
        return None
    return _positive_int(value, name="measured_max_steps")


@dataclass(frozen=True, slots=True)
class MeasuredBudget:
    """Measured whole-run cost, split into incurred and future reserved work."""

    timings: Mapping[str, float]
    prelaunch_timings: Mapping[str, float]
    prelaunch_seconds: float
    endpoint_seconds: float
    reserved_endpoint_seconds: float
    training_seconds_per_step: float
    training_seconds: float
    future_reserved_seconds: float
    future_reserved_seconds_with_headroom: float
    headroom_fraction: float
    headroom_basis_seconds: float
    headroom_seconds: float
    total_seconds: float
    max_steps: int
    measured_cap_steps: int | None
    completion_token_cap: int
    planned_generation_completions: int
    planned_generated_token_cap: int
    disk_actual_bytes: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the measured budget."""
        return cast("dict[str, object]", asdict(self))


def aggregate_measured_budget(
    source: Path | Mapping[str, object],
    *,
    max_steps: int,
) -> MeasuredBudget:
    """Aggregate incurred phases and reserve the remaining bounded experiment cost.

    Headroom applies only to future work.  Already-incurred prelaunch time is measured rather than
    uncertain, and is added after headroom to produce the whole-run ``total_seconds``.
    """
    max_steps = _positive_int(max_steps, name="max_steps")
    payload = _load_json(source)
    (
        completion_token_cap,
        planned_generation_completions,
        planned_generated_token_cap,
        disk_actual_bytes,
    ) = _prelaunch_contract(payload)
    throughput = payload.get("throughput", payload)
    if not isinstance(throughput, Mapping):
        raise TypeError("measurement artifact throughput section must be a JSON object")
    phase_timings = {
        name: _timing_from_payload(cast("Mapping[str, object]", throughput), name)
        for name in REQUIRED_THROUGHPUT_TIMINGS
    }
    prelaunch_timings = _prelaunch_timings(payload)
    endpoint_timings = _endpoint_timings(payload)
    fraction, explicit_headroom = _headroom(payload)
    timings = {**phase_timings, **endpoint_timings}
    train_per_step = sum(phase_timings.values())
    prelaunch_seconds = sum(prelaunch_timings.values())
    endpoint_seconds = sum(endpoint_timings.values())
    training_seconds = train_per_step * max_steps
    future_reserved_seconds = endpoint_seconds + training_seconds
    headroom_seconds = explicit_headroom or future_reserved_seconds * fraction
    future_reserved_seconds_with_headroom = future_reserved_seconds + headroom_seconds
    measured_cap = _measured_cap(payload)
    if max_steps > INITIAL_MAX_STEPS:
        if measured_cap is None:
            raise ValueError(
                f"max_steps={max_steps} exceeds {INITIAL_MAX_STEPS}; a persisted measured cap is required"
            )
        if max_steps > measured_cap:
            raise ValueError(f"max_steps={max_steps} exceeds persisted measured cap {measured_cap}")
    return MeasuredBudget(
        timings=timings,
        prelaunch_timings=prelaunch_timings,
        prelaunch_seconds=prelaunch_seconds,
        endpoint_seconds=endpoint_seconds,
        reserved_endpoint_seconds=endpoint_seconds,
        training_seconds_per_step=train_per_step,
        training_seconds=training_seconds,
        future_reserved_seconds=future_reserved_seconds,
        future_reserved_seconds_with_headroom=future_reserved_seconds_with_headroom,
        headroom_fraction=fraction,
        headroom_basis_seconds=future_reserved_seconds,
        headroom_seconds=headroom_seconds,
        total_seconds=prelaunch_seconds + future_reserved_seconds_with_headroom,
        max_steps=max_steps,
        measured_cap_steps=measured_cap,
        completion_token_cap=completion_token_cap,
        planned_generation_completions=planned_generation_completions,
        planned_generated_token_cap=planned_generated_token_cap,
        disk_actual_bytes=disk_actual_bytes,
    )


def _require_close(actual: float, expected: float, *, name: str) -> None:
    if not math.isclose(actual, expected):
        raise ValueError(f"{name} must equal {expected}, got {actual}")


def _validate_aggregated_budget(payload: Mapping[str, object]) -> None:
    """Validate a persisted aggregate, including its redundant arithmetic receipts."""
    _prelaunch_contract(payload)
    timings = payload.get("timings")
    prelaunch = payload.get("prelaunch_timings")
    if not isinstance(timings, Mapping) or not isinstance(prelaunch, Mapping):
        raise TypeError("aggregated budget needs timings and prelaunch_timings objects")
    missing_timings = (set(REQUIRED_MEASUREMENTS) - {"headroom"}) - set(timings)
    missing_prelaunch = set(REQUIRED_PRELAUNCH_TIMINGS) - set(prelaunch)
    if missing_timings or missing_prelaunch:
        raise ValueError(
            f"aggregated budget is missing timings: {sorted(missing_timings | missing_prelaunch)}"
        )
    timing_values = {
        name: _positive_number(timings[name], name=name)
        for name in set(REQUIRED_MEASUREMENTS) - {"headroom"}
    }
    prelaunch_values = {
        name: _positive_number(prelaunch[name], name=name) for name in REQUIRED_PRELAUNCH_TIMINGS
    }
    max_steps = _positive_int(payload.get("max_steps"), name="max_steps")
    measured_cap = payload.get("measured_cap_steps")
    if max_steps > INITIAL_MAX_STEPS and (
        measured_cap is None or max_steps > _positive_int(measured_cap, name="measured_cap_steps")
    ):
        raise ValueError("aggregated budget exceeds its persisted measured step cap")

    prelaunch_seconds = _positive_number(payload.get("prelaunch_seconds"), name="prelaunch_seconds")
    training_per_step = _positive_number(
        payload.get("training_seconds_per_step"), name="training_seconds_per_step"
    )
    training_seconds = _positive_number(payload.get("training_seconds"), name="training_seconds")
    reserved_endpoints = _positive_number(
        payload.get("reserved_endpoint_seconds"), name="reserved_endpoint_seconds"
    )
    future_reserved = _positive_number(
        payload.get("future_reserved_seconds"), name="future_reserved_seconds"
    )
    headroom_basis = _positive_number(
        payload.get("headroom_basis_seconds"), name="headroom_basis_seconds"
    )
    headroom_seconds = _positive_number(payload.get("headroom_seconds"), name="headroom_seconds")
    future_with_headroom = _positive_number(
        payload.get("future_reserved_seconds_with_headroom"),
        name="future_reserved_seconds_with_headroom",
    )
    total_seconds = _positive_number(payload.get("total_seconds"), name="total_seconds")

    _require_close(prelaunch_seconds, sum(prelaunch_values.values()), name="prelaunch_seconds")
    _require_close(
        training_per_step,
        sum(timing_values[name] for name in REQUIRED_THROUGHPUT_TIMINGS),
        name="training_seconds_per_step",
    )
    _require_close(training_seconds, training_per_step * max_steps, name="training_seconds")
    _require_close(
        reserved_endpoints,
        sum(timing_values[name] for name in REQUIRED_ENDPOINT_TIMINGS),
        name="reserved_endpoint_seconds",
    )
    _require_close(
        future_reserved, training_seconds + reserved_endpoints, name="future_reserved_seconds"
    )
    _require_close(headroom_basis, future_reserved, name="headroom_basis_seconds")
    _require_close(
        future_with_headroom,
        future_reserved + headroom_seconds,
        name="future_reserved_seconds_with_headroom",
    )
    _require_close(total_seconds, prelaunch_seconds + future_with_headroom, name="total_seconds")


def measured_budget_is_complete(source: Path | Mapping[str, object] | MeasuredBudget) -> bool:
    """Return whether a persisted or already-aggregated budget has every required positive term."""
    try:
        payload = asdict(source) if isinstance(source, MeasuredBudget) else _load_json(source)
        if "timings" in payload:
            _validate_aggregated_budget(payload)
        else:
            aggregate_measured_budget(payload, max_steps=INITIAL_MAX_STEPS)
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_measured_budget(path: Path, budget: MeasuredBudget) -> None:
    """Persist an aggregated measured budget with a complete atomic replacement."""
    _write_json(path, budget.as_dict())


def build_parser() -> argparse.ArgumentParser:
    """Expose the CPU ``plan`` and measured ``aggregate`` commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("plan", "aggregate"))
    parser.add_argument("--measurements", type=Path, required=False)
    parser.add_argument("--max-steps", type=int, default=INITIAL_MAX_STEPS)
    parser.add_argument("--out", type=Path, required=False)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--training-corpus", type=Path)
    parser.add_argument("--behavior-manifest", type=Path)
    parser.add_argument("--survey-data-dir", type=Path)
    parser.add_argument("--construct-stimuli", type=Path, action="append", default=[])
    parser.add_argument("--lens-stimuli", type=Path, action="append", default=[])
    parser.add_argument("--model-metadata", type=Path)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--oversample", type=int, default=1)
    parser.add_argument("--completion-tokens", type=int, default=32_768)
    parser.add_argument("--capture-states", type=int, default=DEFAULT_CAPTURE_STATES)
    parser.add_argument("--capture-poolings", type=int, default=DEFAULT_CAPTURE_POOLINGS)
    parser.add_argument("--capture-boundaries", type=int, default=DEFAULT_CAPTURE_BOUNDARIES)
    parser.add_argument("--capture-layers", type=int)
    parser.add_argument("--capture-hidden-size", type=int)
    parser.add_argument("--capture-dtype-bytes", type=int, default=DEFAULT_CAPTURE_DTYPE_BYTES)
    parser.add_argument("--lens-states", type=int, default=DEFAULT_LENS_STATES)
    parser.add_argument("--lens-dtype-bytes", type=int, default=DEFAULT_LENS_DTYPE_BYTES)
    parser.add_argument(
        "--accumulator-dtype-bytes", type=int, default=DEFAULT_ACCUMULATOR_DTYPE_BYTES
    )
    parser.add_argument("--checkpoint-count", type=int)
    parser.add_argument("--intervention-prompts", type=int, default=8)
    parser.add_argument("--intervention-conditions", type=int, default=3)
    parser.add_argument("--intervention-samples", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build a static runtime plan or aggregate persisted measurements."""
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or ("aggregate" if args.measurements is not None else "plan")
    if command == "aggregate":
        if args.measurements is None:
            parser.error("aggregate requires --measurements")
        measured = aggregate_measured_budget(args.measurements, max_steps=args.max_steps)
        if args.out is None:
            print(json.dumps(measured.as_dict(), indent=2, sort_keys=True))  # noqa: T201
        else:
            write_measured_budget(args.out, measured)
        return 0

    required = {
        "--training-manifest": args.training_manifest,
        "--behavior-manifest": args.behavior_manifest,
        "--survey-data-dir": args.survey_data_dir,
        "--model-metadata": args.model_metadata,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if not args.construct_stimuli:
        missing.append("--construct-stimuli")
    if not args.lens_stimuli:
        missing.append("--lens-stimuli")
    if missing:
        parser.error(f"plan requires {', '.join(missing)}")
    metadata_payload = json.loads(args.model_metadata.read_text(encoding="utf-8"))
    if not isinstance(metadata_payload, Mapping):
        parser.error("--model-metadata must point to a JSON object")
    plan = plan_budget(
        training_manifest=args.training_manifest,
        training_corpus=args.training_corpus,
        behavior_manifest=args.behavior_manifest,
        survey_data_dir=args.survey_data_dir,
        construct_stimuli=args.construct_stimuli,
        lens_stimuli=args.lens_stimuli,
        model=cast("Mapping[str, object]", metadata_payload),
        max_steps=args.max_steps,
        prompts_per_step=args.prompts_per_step,
        group_size=args.group_size,
        oversample=args.oversample,
        completion_tokens=args.completion_tokens,
        capture_states=args.capture_states,
        capture_poolings=args.capture_poolings,
        capture_boundaries=args.capture_boundaries,
        capture_layers=args.capture_layers,
        capture_hidden_size=args.capture_hidden_size,
        capture_dtype_bytes=args.capture_dtype_bytes,
        lens_states=args.lens_states,
        lens_dtype_bytes=args.lens_dtype_bytes,
        accumulator_dtype_bytes=args.accumulator_dtype_bytes,
        checkpoint_count=args.checkpoint_count,
        intervention_prompts=args.intervention_prompts,
        intervention_conditions=args.intervention_conditions,
        intervention_samples=args.intervention_samples,
    )
    if args.out is None:
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))  # noqa: T201
    else:
        _write_json(args.out, plan.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
