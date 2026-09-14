"""Run the frozen cooperation evaluation endpoints.

The endpoint plans live in :mod:`games.cooperation_evals`; this module only selects a plan,
resolves the serving path, and hands the rendered requests to the native resumable battery.  The
``--print-plan`` path renders and counts requests on CPU and never resolves a model or backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from reward_hacking.model_backend import Backend

from games import cooperation_evals
from games.eval_model import (
    ServedModel,
    resolve_served_model,
    sha256_of_file,
    verify_served_model,
)
from games.eval_sampler import (
    DEFAULT_EVAL_MAX_NEW_TOKENS,
    DEFAULT_SAMPLER_MODE,
    add_sampler_arg,
    eval_sampling,
    resolve_sampler_mode,
)
from games.evals import (
    ADMISSION_LONGEST_FIRST,
    ADMISSIONS,
    SECTION_DT_PROBES,
    SECTION_GAME_BEHAVIOR,
    SECTION_SELF_REPORT,
    SUBMISSION_POOLED,
    SUBMISSIONS,
    EvalConfig,
    PlannedRequest,
    rebuild_summary,
    run_eval_battery,
)
from games.interp_lens_ladder import adapter_digests
from games.provenance import git_sha
from games.survey import FAMILY_SELF_PREDICTION, battery_orders, survey_battery
from reward_hacking import backend_cli

logger = logging.getLogger(__name__)

ENDPOINT_BEHAVIOR = "behavior"
ENDPOINT_ALLOCATION = "allocation"
ENDPOINT_FULL_CONTEXT = "full-context"
ENDPOINT_CORE_SURVEY = "core-survey"
ENDPOINT_PROSOCIALNESS = "prosocialness"
ENDPOINT_LOCAL_DT = "local-dt"
ENDPOINTS: tuple[str, ...] = (
    ENDPOINT_BEHAVIOR,
    ENDPOINT_ALLOCATION,
    ENDPOINT_FULL_CONTEXT,
    ENDPOINT_CORE_SURVEY,
    ENDPOINT_PROSOCIALNESS,
    ENDPOINT_LOCAL_DT,
)
ENDPOINT_ALL = "all"
ENDPOINT_CHOICES: tuple[str, ...] = (*ENDPOINTS, ENDPOINT_ALL)
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_ARM = "cooperation-generalization-care-alpha-1"
DEFAULT_COMPLETION_TOKENS = 32_768
DEFAULT_BEHAVIOR_MANIFEST = cooperation_evals.DEFAULT_BEHAVIOR_MANIFEST_PATH
DEFAULT_SURVEY_DATA_DIR = Path("games/data/survey")
DEFAULT_OUT_DIR = Path("artifacts/games/cooperation-generalization/evaluations")
PROFILE_RESEARCH = "research"
PROFILE_SERVING_SMOKE = "serving-smoke"
PROFILE_CHOICES: tuple[str, ...] = (PROFILE_RESEARCH, PROFILE_SERVING_SMOKE)
SERVING_SMOKE_SELF_PREDICTION = "serving-smoke-self-prediction"
SERVING_SMOKE_FORECAST = "serving-smoke-forecast"
SERVING_SMOKE_ENDPOINTS: tuple[str, ...] = (
    SERVING_SMOKE_SELF_PREDICTION,
    SERVING_SMOKE_FORECAST,
)

DEFAULT_CONTEXT_SAMPLES = 2
DEFAULT_PREFILLED_THINK = True
DEFAULT_THINKING = True
SUMMARY_SUFFIX = ".summary.json"
TRACE_SUFFIX = ".jsonl"

# A mock run deliberately produces unparseable records.  It is useful for checking the trace and
# resume plumbing, but the resulting summaries must never read as behavioral evidence.
MOCK_RESPONSES: tuple[str, ...] = ("synthetic mock completion; no answer tags",)


def _model_source_for_execution(args: argparse.Namespace) -> str:
    """Return the explicit local load source, refusing an execution without one."""
    source = args.base_model or args.model
    if source is None:
        raise ValueError("--model is required when executing cooperation endpoints")
    return str(source)


@dataclass(frozen=True, slots=True)
class EndpointSpec:
    """The section and output identity for one endpoint."""

    name: str
    sections: tuple[str, ...]


ENDPOINT_SPECS: dict[str, EndpointSpec] = {
    ENDPOINT_BEHAVIOR: EndpointSpec(ENDPOINT_BEHAVIOR, (SECTION_GAME_BEHAVIOR,)),
    ENDPOINT_ALLOCATION: EndpointSpec(ENDPOINT_ALLOCATION, (SECTION_SELF_REPORT,)),
    ENDPOINT_FULL_CONTEXT: EndpointSpec(ENDPOINT_FULL_CONTEXT, (SECTION_SELF_REPORT,)),
    ENDPOINT_CORE_SURVEY: EndpointSpec(ENDPOINT_CORE_SURVEY, (SECTION_SELF_REPORT,)),
    ENDPOINT_PROSOCIALNESS: EndpointSpec(ENDPOINT_PROSOCIALNESS, (SECTION_SELF_REPORT,)),
    ENDPOINT_LOCAL_DT: EndpointSpec(ENDPOINT_LOCAL_DT, (SECTION_DT_PROBES,)),
    SERVING_SMOKE_SELF_PREDICTION: EndpointSpec(
        SERVING_SMOKE_SELF_PREDICTION, (SECTION_SELF_REPORT,)
    ),
    SERVING_SMOKE_FORECAST: EndpointSpec(SERVING_SMOKE_FORECAST, (SECTION_SELF_REPORT,)),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the endpoint CLI without loading a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        choices=ENDPOINT_CHOICES,
        default=ENDPOINT_ALL,
        help=(
            "Evaluation cell to run; all runs the six research cells or the two cells owned by "
            "--evaluation-profile serving-smoke in one backend session."
        ),
    )
    parser.add_argument(
        "--evaluation-profile",
        dest="evaluation_profile",
        choices=PROFILE_CHOICES,
        default=PROFILE_RESEARCH,
        help=(
            "Research runs all requested frozen cells. serving-smoke runs only the historical "
            "self-prediction and exact-context forecast seams with one draw per order."
        ),
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Render exact request counts and identities on CPU, without loading a model.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_BEHAVIOR_MANIFEST,
        help="Private behavior roster manifest.",
    )
    parser.add_argument(
        "--survey-data-dir",
        type=Path,
        default=DEFAULT_SURVEY_DATA_DIR,
        help="Runtime directory containing the survey item JSON files.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Immutable local model snapshot used at execution time; sequence passes this explicitly.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Canonical model identity used for sampler and trace provenance.",
    )
    parser.add_argument(
        "--checkpoint",
        "--adapter",
        dest="checkpoint",
        type=Path,
        default=None,
        help="LoRA checkpoint to serve on top of --model; vLLM keeps it unmerged.",
    )
    parser.add_argument("--base-model", dest="base_model", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--arm", default=DEFAULT_ARM, help="Arm label recorded in endpoint metadata."
    )
    parser.add_argument(
        "--step", type=int, default=0, help="Checkpoint step label recorded in endpoint metadata."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory containing one trace and summary per endpoint.",
    )
    parser.add_argument(
        "--context-samples",
        type=int,
        default=DEFAULT_CONTEXT_SAMPLES,
        help="Samples per printed order for exact-context items; the research cell is frozen at 2.",
    )
    parser.add_argument(
        "--trained-game-id",
        dest="trained_game_ids",
        action="append",
        default=[],
        help="Training game identity to carry into behavior records; repeat for multiple games.",
    )
    parser.add_argument(
        "--rollout-id",
        default="cooperation-eval",
        help="Identity of this rendered behavior rollout.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Requested local generation batch width recorded in the endpoint config.",
    )
    parser.add_argument(
        "--prefilled-think",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PREFILLED_THINK,
        help="Whether the local chat template prefilled the opening thinking tag.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue an incomplete trace by native request identity (default: true).",
    )
    parser.add_argument("--submission", choices=SUBMISSIONS, default=SUBMISSION_POOLED)
    parser.add_argument("--admission", choices=ADMISSIONS, default=ADMISSION_LONGEST_FIRST)
    backend_cli.add_backend_args(parser, default="vllm")
    add_sampler_arg(parser)
    # The endpoint battery is intentionally pinned to the policy that generated the training data;
    # keeping this visible in parsed arguments also lets the sequence command be checked by the
    # native parser before a GPU is reserved.
    parser.set_defaults(
        sampler=DEFAULT_SAMPLER_MODE,
        max_new_tokens=DEFAULT_EVAL_MAX_NEW_TOKENS,
        thinking=DEFAULT_THINKING,
    )
    return parser


def _check_required_path(path: Path, *, label: str, directory: bool = False) -> None:
    """Refuse a missing runtime input before a plan claims to be complete."""
    if directory:
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory is missing: {path}")
    elif not path.is_file():
        raise FileNotFoundError(f"{label} file is missing: {path}")


def _selected_endpoints(args: argparse.Namespace) -> tuple[str, ...]:
    """Resolve the explicit profile into its immutable endpoint artifact names."""
    profile = str(args.evaluation_profile)
    if profile == PROFILE_SERVING_SMOKE:
        if args.endpoint != ENDPOINT_ALL:
            raise ValueError(
                f"--evaluation-profile {PROFILE_SERVING_SMOKE!r} owns its two smoke cells; omit "
                f"--endpoint or pass --endpoint {ENDPOINT_ALL!r}"
            )
        return SERVING_SMOKE_ENDPOINTS
    if profile == PROFILE_RESEARCH:
        return ENDPOINTS if args.endpoint == ENDPOINT_ALL else (str(args.endpoint),)
    raise ValueError(f"unknown cooperation profile {profile!r}")


def validate_print_plan_inputs(*, endpoint: str, manifest: Path, survey_data_dir: Path) -> None:
    """Validate only the runtime data needed by the selected CPU plan."""
    selected = (
        SERVING_SMOKE_ENDPOINTS
        if endpoint == PROFILE_SERVING_SMOKE
        else ENDPOINTS
        if endpoint == ENDPOINT_ALL
        else (endpoint,)
    )
    if any(
        name in selected
        for name in (
            ENDPOINT_BEHAVIOR,
            ENDPOINT_ALLOCATION,
            ENDPOINT_FULL_CONTEXT,
            SERVING_SMOKE_FORECAST,
        )
    ):
        _check_required_path(manifest, label="behavior manifest")
    if any(
        name in selected
        for name in (ENDPOINT_CORE_SURVEY, ENDPOINT_PROSOCIALNESS, SERVING_SMOKE_SELF_PREDICTION)
    ):
        _check_required_path(survey_data_dir, label="survey data", directory=True)


def _manifest_digest(path: Path) -> str:
    _check_required_path(path, label="behavior manifest")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_digest(path: Path) -> str:
    """Digest runtime item bytes with relative names, independent of its machine-local root."""
    _check_required_path(path, label="survey data", directory=True)
    parts = [
        f"{child.relative_to(path).as_posix()}:{hashlib.sha256(child.read_bytes()).hexdigest()}"
        for child in sorted(path.rglob("*"))
        if child.is_file()
    ]
    if not parts:
        raise ValueError(f"survey data directory {path} contains no files")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def rendered_prompt_digest(requests: Sequence[PlannedRequest]) -> str:
    """Digest request identities and rendered prompt bytes without publishing prompt text."""
    payload = [
        {"identity": list(request.identity), "prompt": request.prompt} for request in requests
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rendered_prompt_count(requests: Sequence[PlannedRequest]) -> int:
    """Count unique rendered prompts, excluding repeated sample draws."""
    return len({(request.section, request.identity[:-1], request.prompt) for request in requests})


def _self_prediction_smoke_plan(
    args: argparse.Namespace,
) -> tuple[PlannedRequest, ...]:
    """Select every current self-prediction item in both orders at one sample index."""
    full_core = cooperation_evals.build_core_survey_plan(
        data_dir=args.survey_data_dir,
        samples=2,
        prefilled_think=bool(args.prefilled_think),
    )
    items = survey_battery(
        data_dir=args.survey_data_dir,
        families=(FAMILY_SELF_PREDICTION,),
        instruments=(),
        tier="core",
    )
    item_ids = {item.item_id for item in items}
    expected_pairs = {
        (item.item_id, order_name)
        for item in items
        for order_name, _option_order in (battery_orders(item) or (("not-applicable", ()),))
    }
    selected = tuple(
        request
        for request in full_core
        if request.identity[1] in item_ids and request.identity[-1] == 0
    )
    actual_pairs = {(request.identity[1], request.identity[2]) for request in selected}
    if actual_pairs != expected_pairs or len(selected) != len(expected_pairs):
        raise ValueError(
            "serving smoke self-prediction membership drifted: the rendered plan does not contain "
            "each current self-prediction item under each current presentation order"
        )
    return selected


def _forecast_smoke_plan(
    cooperation_plan: cooperation_evals.CooperationEvalPlan,
) -> tuple[PlannedRequest, ...]:
    """Select every exact-context forecast item at one sample index."""
    selected = tuple(request for request in cooperation_plan.forecasts if request.identity[-1] == 0)
    expected_items = {
        (request.identity[1], request.identity[2]) for request in cooperation_plan.forecasts
    }
    actual_items = {(request.identity[1], request.identity[2]) for request in selected}
    if actual_items != expected_items or len(selected) != len(expected_items):
        raise ValueError(
            "serving smoke forecast membership drifted: the rendered plan does not contain "
            "each exact-context forecast item at one sample index"
        )
    return selected


def _plans_for_endpoint(args: argparse.Namespace) -> dict[str, tuple[PlannedRequest, ...]]:
    """Build the selected endpoint plans, before any backend exists."""
    selected = _selected_endpoints(args)
    plans: dict[str, tuple[PlannedRequest, ...]] = {}
    cooperation_plan: cooperation_evals.CooperationEvalPlan | None = None

    def get_cooperation_plan() -> cooperation_evals.CooperationEvalPlan:
        nonlocal cooperation_plan
        if cooperation_plan is None:
            cooperation_plan = cooperation_evals.build_cooperation_plan(
                args.manifest,
                context_samples=args.context_samples,
                prefilled_think=bool(args.prefilled_think),
                trained_game_ids=tuple(args.trained_game_ids),
                rollout_id=str(args.rollout_id),
            )
        return cooperation_plan

    def core_survey() -> tuple[PlannedRequest, ...]:
        return tuple(
            cooperation_evals.build_core_survey_plan(
                data_dir=args.survey_data_dir,
                samples=2,
                prefilled_think=bool(args.prefilled_think),
            )
        )

    def prosocialness() -> tuple[PlannedRequest, ...]:
        return tuple(
            cooperation_evals.build_prosocialness_plan(
                data_dir=args.survey_data_dir,
                samples=2,
                prefilled_think=bool(args.prefilled_think),
            )
        )

    def local_dt() -> tuple[PlannedRequest, ...]:
        return tuple(
            cooperation_evals.build_local_dt_plan(
                multiple_choice_samples=2,
                open_ended_samples=2,
                prefilled_think=bool(args.prefilled_think),
            )
        )

    plan_builders: dict[str, Callable[[], tuple[PlannedRequest, ...]]] = {
        ENDPOINT_BEHAVIOR: lambda: tuple(get_cooperation_plan().behavior),
        ENDPOINT_ALLOCATION: lambda: tuple(get_cooperation_plan().allocation),
        ENDPOINT_FULL_CONTEXT: lambda: (
            *get_cooperation_plan().forecasts,
            *get_cooperation_plan().normative,
        ),
        ENDPOINT_CORE_SURVEY: core_survey,
        ENDPOINT_PROSOCIALNESS: prosocialness,
        ENDPOINT_LOCAL_DT: local_dt,
        SERVING_SMOKE_SELF_PREDICTION: lambda: _self_prediction_smoke_plan(args),
        SERVING_SMOKE_FORECAST: lambda: _forecast_smoke_plan(get_cooperation_plan()),
    }
    for endpoint in selected:
        try:
            plans[endpoint] = plan_builders[endpoint]()
        except KeyError as error:
            raise ValueError(f"unknown cooperation endpoint {endpoint!r}") from error
    return plans


def _behavior_sample_count(plan: Sequence[PlannedRequest]) -> int:
    """Read the frozen behavior draw count from the supplied request identities."""
    counts_by_prompt: dict[tuple[Any, ...], set[Any]] = {}
    for request in plan:
        counts_by_prompt.setdefault(tuple(request.identity[:-1]), set()).add(request.identity[-1])
    counts = {len(samples) for samples in counts_by_prompt.values()}
    if len(counts) != 1:
        raise ValueError(
            "behavior plan has inconsistent samples per rendered prompt; its EvalConfig cannot "
            f"represent counts {sorted(counts)}"
        )
    return counts.pop()


def _config_for_endpoint(
    args: argparse.Namespace, endpoint: str, *, plan: Sequence[PlannedRequest] | None = None
) -> EvalConfig:
    """Build the config matching the planner for one endpoint."""
    if endpoint == ENDPOINT_BEHAVIOR:
        if plan is None:
            raise ValueError("behavior config requires its explicit request plan")
        config = EvalConfig(
            game_behavior_samples=_behavior_sample_count(plan),
            survey_samples=args.context_samples,
            prefilled_think=bool(args.prefilled_think),
            trained_game_ids=tuple(args.trained_game_ids),
            batch_size=args.batch_size,
        )
    elif endpoint in (ENDPOINT_ALLOCATION, ENDPOINT_FULL_CONTEXT):
        config = EvalConfig(
            survey_samples=args.context_samples,
            prefilled_think=bool(args.prefilled_think),
            batch_size=args.batch_size,
        )
    elif endpoint == ENDPOINT_CORE_SURVEY:
        counts = cooperation_evals.survey_leg_counts("core", data_dir=args.survey_data_dir)
        config = EvalConfig(
            survey_samples=2,
            survey_data_dir=args.survey_data_dir,
            survey_families=counts.families,
            survey_instruments=counts.instruments,
            survey_tier="core",
            prefilled_think=bool(args.prefilled_think),
            batch_size=args.batch_size,
        )
    elif endpoint == ENDPOINT_PROSOCIALNESS:
        counts = cooperation_evals.survey_leg_counts("prosocialness", data_dir=args.survey_data_dir)
        config = EvalConfig(
            survey_samples=2,
            survey_data_dir=args.survey_data_dir,
            survey_families=counts.families,
            survey_instruments=counts.instruments,
            survey_tier="",
            prefilled_think=bool(args.prefilled_think),
            batch_size=args.batch_size,
        )
    elif endpoint == SERVING_SMOKE_SELF_PREDICTION:
        config = EvalConfig(
            survey_samples=1,
            survey_data_dir=args.survey_data_dir,
            survey_families=(FAMILY_SELF_PREDICTION,),
            survey_tier="core",
            prefilled_think=bool(args.prefilled_think),
            batch_size=args.batch_size,
        )
    elif endpoint == SERVING_SMOKE_FORECAST:
        config = EvalConfig(
            survey_samples=1,
            prefilled_think=bool(args.prefilled_think),
            batch_size=args.batch_size,
        )
    elif endpoint == ENDPOINT_LOCAL_DT:
        config = EvalConfig(
            multiple_choice_samples=2,
            open_ended_samples=2,
            dtbench_dir=None,
            prefilled_think=bool(args.prefilled_think),
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(f"unknown cooperation endpoint {endpoint!r}")
    return config


def endpoint_meta(  # noqa: PLR0913 - metadata has one explicit field per identity component
    *,
    endpoint: str,
    manifest_path: Path | None,
    manifest_digest: str | None,
    requests: Sequence[PlannedRequest],
    model_identity: str,
    model_weights_identity: str | None = None,
    adapter_digests: Mapping[str, str | None],
    settings: Mapping[str, Any],
    evaluation_profile: str = PROFILE_RESEARCH,
    served: Mapping[str, Any] | None = None,
    input_digests: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Build caller-owned metadata that joins every trace to its exact input identity."""
    payload: dict[str, Any] = {
        "cooperation_endpoint": endpoint,
        "cooperation_evaluation_profile": evaluation_profile,
        "cooperation_manifest_path": None if manifest_path is None else str(manifest_path),
        "cooperation_manifest_digest": manifest_digest,
        "cooperation_rendered_prompt_digest": rendered_prompt_digest(requests),
        "cooperation_request_count": len(requests),
        "cooperation_rendered_prompt_count": _rendered_prompt_count(requests),
        "cooperation_code_git_sha": git_sha(),
        "model_identity": model_identity,
        "model_weights_identity": model_weights_identity or model_identity,
        "adapter_weights_sha256": adapter_digests.get("weights"),
        "adapter_config_sha256": adapter_digests.get("config"),
        "settings": dict(settings),
        "sampler": settings.get("sampler"),
    }
    if input_digests is not None:
        payload["cooperation_input_digests"] = dict(input_digests)
    if served is not None:
        payload.update(dict(served))
    return payload


def _summary_path(trace_path: Path) -> Path:
    return trace_path.with_suffix(SUMMARY_SUFFIX)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast("dict[str, Any]", payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _complete_cell(
    trace_path: Path, *, expected_meta: Mapping[str, Any] | None = None
) -> dict[str, Any] | None:
    """Return a validated summary for a complete cell, without touching a backend."""
    summary_path = _summary_path(trace_path)
    if not trace_path.exists() and not summary_path.exists():
        return None
    if trace_path.exists() and not summary_path.exists():
        # The native battery owns partial-trace continuation.  It validates the existing meta,
        # skips finished identities, and appends each newly parsed record before returning.
        return None
    if not trace_path.exists():
        raise ValueError(
            f"endpoint cell is incomplete: expected both {trace_path} and {summary_path}; "
            "relaunch the same command with native resume"
        )
    stored = _read_json_object(summary_path)
    trace_lines = trace_path.read_text(encoding="utf-8").splitlines()
    if not trace_lines:
        raise ValueError(f"trace {trace_path} is empty")
    trace_meta_value = json.loads(trace_lines[0])
    if not isinstance(trace_meta_value, dict):
        raise TypeError(f"trace {trace_path} starts with a non-object record")
    trace_meta = cast("dict[str, Any]", trace_meta_value)
    if expected_meta is not None:
        drifted = [
            key
            for key, expected in expected_meta.items()
            if key != "cooperation_code_git_sha" and trace_meta.get(key) != expected
        ]
        if drifted:
            raise ValueError(
                f"complete endpoint {trace_path} has identity drift in {drifted}; "
                "point --out-dir at a fresh directory for a different measurement"
            )
    rebuilt = rebuild_summary(trace_path)
    if stored != rebuilt:
        raise ValueError(
            f"summary {summary_path} disagrees with its trace {trace_path}; refusing to treat a "
            "tampered or stale cell as complete"
        )
    return stored


def _run_native_cell(  # noqa: PLR0913 - each argument is one native-cell decision
    backend: Backend,
    *,
    endpoint: str,
    plan: Sequence[PlannedRequest],
    out_path: Path,
    config: EvalConfig,
    meta: Mapping[str, Any],
    submission: str,
    admission: str,
    resume: bool,
    expected_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one plan through the native per-record evaluator and close its summary."""
    complete = _complete_cell(out_path, expected_meta=expected_meta)
    if complete is not None:
        return complete
    sections = ENDPOINT_SPECS[endpoint].sections
    summary = run_eval_battery(
        backend,
        sections=sections,
        out_path=out_path,
        meta=meta,
        config=config,
        submission=submission,
        admission=admission,
        resume=resume,
        plan=plan,
    )
    _write_json(_summary_path(out_path), summary)
    return summary


def _settings_for(args: argparse.Namespace, endpoint: str) -> dict[str, Any]:
    source = _model_source_for_execution(args)
    return {
        "endpoint": endpoint,
        "arm": args.arm,
        "step": args.step,
        "sampler": args.sampler,
        "max_new_tokens": args.max_new_tokens,
        "thinking": args.thinking,
        "prefilled_think": args.prefilled_think,
        "context_samples": args.context_samples,
        "submission": args.submission,
        "admission": args.admission,
        "backend": args.backend,
        "evaluation_profile": args.evaluation_profile,
        "model_id": args.model_id,
        "model_source": source,
    }


def _model_identity(model: str) -> str:
    """Return a stable model identity, hashing local tensor files when available."""
    candidate = Path(model)
    if not candidate.is_dir():
        return model
    tensors = sorted(candidate.rglob("*.safetensors"))
    if not tensors:
        return f"directory:{candidate.resolve()}"
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.relative_to(candidate).as_posix().encode("utf-8"))
        digest.update(sha256_of_file(tensor).encode("utf-8"))
    return f"weights:{digest.hexdigest()}"


def _serving_and_backend(
    args: argparse.Namespace, *, out_dir: Path
) -> tuple[Backend, ServedModel, dict[str, str | None]]:
    """Resolve one backend and preserve the vLLM runtime-adapter serving decision."""
    model = _model_source_for_execution(args)
    served = resolve_served_model(
        checkpoint=args.checkpoint,
        base_model=str(args.model_id),
        base_model_source=model,
        backend_kind=args.backend,
        merge_root=out_dir / "merged",
        merge_label=f"{args.arm}-step-{args.step}-",
    )
    hashes: dict[str, str | None] = {"weights": None, "config": None}
    if args.checkpoint is not None:
        weights_sha, config_sha = adapter_digests(args.checkpoint)
        hashes = {"weights": weights_sha, "config": config_sha}
    sampling = eval_sampling(resolve_sampler_mode(args), thinking=bool(args.thinking))
    backend = backend_cli.backend_from_args(
        args,
        served.model_id,
        local_sampling=sampling,
        mock_responses=MOCK_RESPONSES,
        extra_kwargs=served.backend_kwargs,
    )
    verify_served_model(backend, served)
    return backend, served, hashes


def _endpoint_identity_meta(  # noqa: PLR0913 - identity has one explicit field per input
    args: argparse.Namespace,
    *,
    endpoint: str,
    requests: Sequence[PlannedRequest],
    model_identity: str,
    model_weights_identity: str,
    adapter_hashes: Mapping[str, str | None],
) -> dict[str, Any]:
    """Build the model-free identity fields used to validate a complete cell."""
    manifest_path: Path | None = (
        args.manifest
        if endpoint
        in {
            ENDPOINT_BEHAVIOR,
            ENDPOINT_ALLOCATION,
            ENDPOINT_FULL_CONTEXT,
            SERVING_SMOKE_FORECAST,
        }
        else None
    )
    manifest_hash = _manifest_digest(args.manifest) if manifest_path is not None else None
    input_hashes: dict[str, str | None] = {}
    if manifest_path is not None:
        input_hashes["behavior_manifest"] = manifest_hash
    if endpoint in (
        ENDPOINT_CORE_SURVEY,
        ENDPOINT_PROSOCIALNESS,
        SERVING_SMOKE_SELF_PREDICTION,
    ):
        input_hashes["survey_data"] = _directory_digest(args.survey_data_dir)
    payload = endpoint_meta(
        endpoint=endpoint,
        manifest_path=manifest_path,
        manifest_digest=manifest_hash,
        requests=requests,
        model_identity=model_identity,
        model_weights_identity=model_weights_identity,
        adapter_digests=adapter_hashes,
        settings=_settings_for(args, endpoint),
        evaluation_profile=args.evaluation_profile,
        input_digests=input_hashes,
    )
    if manifest_path is not None:
        roster = cooperation_evals.expand_behavior_roster(
            cooperation_evals.load_behavior_manifest(manifest_path)
        )
        payload["cooperation_scenario_group_ids"] = sorted(roster.scenario_group_ids)
    if endpoint in SERVING_SMOKE_ENDPOINTS:
        payload["cooperation_smoke_membership"] = {
            "profile": PROFILE_SERVING_SMOKE,
            "samples": 1,
            "item_count": len({request.identity[1] for request in requests}),
            "request_count": len(requests),
            "order_names": sorted({str(request.identity[2]) for request in requests}),
            "family": (
                FAMILY_SELF_PREDICTION
                if endpoint == SERVING_SMOKE_SELF_PREDICTION
                else cooperation_evals.INSTRUMENT_CONTEXT_SELF_PREDICTION
            ),
        }
    return payload


def _validate_frozen_settings(args: argparse.Namespace) -> None:
    """Refuse settings that would change the registered cooperation measurement."""
    if args.context_samples != DEFAULT_CONTEXT_SAMPLES:
        raise ValueError("--context-samples is frozen at 2 for the cooperation battery")
    if args.max_new_tokens != DEFAULT_COMPLETION_TOKENS:
        raise ValueError(
            f"--max-new-tokens is frozen at {DEFAULT_COMPLETION_TOKENS} for research endpoints"
        )
    if args.sampler != DEFAULT_SAMPLER_MODE:
        raise ValueError(
            f"--sampler is frozen at {DEFAULT_SAMPLER_MODE!r} for the training-distribution battery"
        )
    if args.evaluation_profile == PROFILE_SERVING_SMOKE:
        if args.backend != "vllm":
            raise ValueError("serving smoke is frozen to the vLLM backend")
        if args.thinking is not True:
            raise ValueError("serving smoke is frozen with thinking enabled")


def run_endpoints(
    args: argparse.Namespace, *, backend: Backend | None = None
) -> dict[str, dict[str, Any]]:
    """Run selected cells, reusing one backend and skipping validated complete cells."""
    _validate_frozen_settings(args)
    selected = _selected_endpoints(args)
    plans = _plans_for_endpoint(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    serving: ServedModel | None = None
    adapter_hashes: dict[str, str | None] = {"weights": None, "config": None}
    if args.checkpoint is not None:
        weights_sha, config_sha = adapter_digests(args.checkpoint)
        adapter_hashes = {"weights": weights_sha, "config": config_sha}
    model_source_path = _model_source_for_execution(args)
    model_identity = str(args.model_id)
    model_weights_identity = _model_identity(model_source_path)
    for endpoint in selected:
        out_path = out_dir / f"{endpoint}{TRACE_SUFFIX}"
        identity_meta = _endpoint_identity_meta(
            args,
            endpoint=endpoint,
            requests=plans[endpoint],
            model_identity=model_identity,
            model_weights_identity=model_weights_identity,
            adapter_hashes=adapter_hashes,
        )
        complete = _complete_cell(out_path, expected_meta=identity_meta)
        if complete is not None:
            results[endpoint] = complete
            continue
        if backend is None:
            resolved_backend, serving, adapter_hashes = _serving_and_backend(args, out_dir=out_dir)
            backend = resolved_backend
        meta = {
            **identity_meta,
            **({} if serving is None else serving.provenance),
        }
        config = _config_for_endpoint(args, endpoint, plan=plans[endpoint])
        results[endpoint] = _run_native_cell(
            backend,
            endpoint=endpoint,
            plan=plans[endpoint],
            out_path=out_path,
            config=config,
            meta=meta,
            submission=args.submission,
            admission=args.admission,
            resume=bool(args.resume),
            expected_meta=meta,
        )
    return results


def print_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build and return exact endpoint counts without constructing a model or backend."""
    _validate_frozen_settings(args)
    selected = _selected_endpoints(args)
    validate_print_plan_inputs(
        endpoint=(
            PROFILE_SERVING_SMOKE
            if args.evaluation_profile == PROFILE_SERVING_SMOKE
            else str(args.endpoint)
        ),
        manifest=args.manifest,
        survey_data_dir=args.survey_data_dir,
    )
    plans = _plans_for_endpoint(args)
    endpoints: dict[str, dict[str, Any]] = {}
    for endpoint in selected:
        plan = plans[endpoint]
        endpoint_counts: dict[str, Any] = {
            "requests": len(plan),
            "responses": len(plan),
            "rendered_prompts": _rendered_prompt_count(plan),
            "rendered_prompt_digest": rendered_prompt_digest(plan),
        }
        if endpoint == ENDPOINT_FULL_CONTEXT:
            endpoint_counts.update(
                {
                    "forecast_requests": sum(
                        request.call_group == cooperation_evals.INSTRUMENT_CONTEXT_SELF_PREDICTION
                        for request in plan
                    ),
                    "normative_requests": sum(
                        request.call_group == cooperation_evals.INSTRUMENT_NORMATIVE_PAYOFF
                        for request in plan
                    ),
                }
            )
        if endpoint in SERVING_SMOKE_ENDPOINTS:
            endpoint_counts["smoke_membership"] = {
                "profile": PROFILE_SERVING_SMOKE,
                "samples": 1,
                "item_count": len({request.identity[1] for request in plan}),
                "request_count": len(plan),
                "order_names": sorted({str(request.identity[2]) for request in plan}),
                "family": (
                    FAMILY_SELF_PREDICTION
                    if endpoint == SERVING_SMOKE_SELF_PREDICTION
                    else cooperation_evals.INSTRUMENT_CONTEXT_SELF_PREDICTION
                ),
            }
        endpoints[endpoint] = endpoint_counts
    return {
        "endpoint": args.endpoint,
        "evaluation_profile": args.evaluation_profile,
        "endpoints": endpoints,
        "total_requests": sum(int(value["requests"]) for value in endpoints.values()),
        "settings": {
            "backend": args.backend,
            "sampler": args.sampler,
            "max_new_tokens": args.max_new_tokens,
            "context_samples": args.context_samples,
            "prefilled_think": args.prefilled_think,
            "thinking": args.thinking,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print a CPU plan or execute the selected resumable endpoint cells."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = build_parser().parse_args(argv)
    if args.print_plan:
        sys.stdout.write(json.dumps(print_plan(args), indent=2, sort_keys=True) + "\n")
        return 0
    run_endpoints(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
