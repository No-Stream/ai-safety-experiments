"""Bounded base/final Jacobian-lens read for cooperation generalization.

This is deliberately narrower than :mod:`games.interp_lens_ladder`: it fits exactly two
model-specific lenses (the unadapted base and one final unmerged adapter), keeps the ten-prompt
smoke and fifty-prompt measurement corpora under different identities, and scores each lens on
eight disjoint prompts.  The base lens is additionally applied to the final model as an explicitly
approximate shared-coordinate sensitivity check.

The prompt files use the existing ``games.interp_cells.Stimulus`` JSONL contract. Each ``pair_id``
must contain exactly two contrasting sides; ``metadata.scenario_group`` keeps whole related
scenario/template families in either fit or quality, never both. Both files are private runtime
inputs under ``docs/scratch`` or ``artifacts``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import AutoTokenizer

from games.deltanet_kernels import (
    DELTANET_KERNEL_FIELD,
    bound_deltanet_kernels,
    bridge_and_check_decode_kernel,
    prefill_deltanet_kernels,
)
from games.interp_capture import LadderCell, render_stimuli
from games.interp_cells import (
    BASE_ARM,
    BASE_STEP,
    Stimulus,
    construct_group,
    load_ladder,
    load_stimuli,
    sha256_of_file,
    stimuli_digest,
)
from games.interp_lens_ladder import (
    DEFAULT_EVAL_POSITIONS,
    DEFAULT_TOP_K,
    LENS_FILENAME,
    LENS_MODEL_UNMERGED,
    LadderLensModels,
    adapter_digests,
    lens_cache_key_for_cell,
    pair_divergence_indices,
)
from games.lora import assert_one_adapter_config
from games.provenance import git_sha
from games.tokenizer_identity import tokenizer_content_sha256
from reward_hacking.interp.directions import matched_norm_random_direction
from reward_hacking.interp.jacobian import (
    DEFAULT_MAX_SEQ_LEN_CEILING,
    JLENS_COMMIT,
    JLENS_UNMERGED_LOAD_PATH,
    JacobianConfig,
    LensCache,
    LensCacheKey,
    _require_jlens,  # pyright: ignore[reportPrivateUsage]
    acquire_lens,
    derive_max_seq_len,
    digest_strings,
    evaluate_reconstruction,
    fit_quality_payload,
    fit_skip_first,
    resolve_weights_identity,
    transport_and_decode,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PRIVATE_STIMULUS_ROOTS = (Path("docs/scratch"), Path("artifacts"))
REPORT_FILENAME = "cooperation_lens.json"
ACCUMULATION_FILENAME = "accumulator.pt"
ACCUMULATION_IDENTITY_SUFFIX = ".identity.json"

PROFILE_SMOKE = "smoke"
PROFILE_MEASUREMENT = "measurement"
PROFILE_FIT_PROMPTS = {PROFILE_SMOKE: 10, PROFILE_MEASUREMENT: 50}
N_QUALITY_PROMPTS = 8

STATE_BASE = "base"
STATE_FINAL = "final"
STATES = (STATE_BASE, STATE_FINAL)

DIRECTION_COSTLY_OTHER_REGARD = "costly-other-regard"
DIRECTION_DECISION_DEPENDENCE = "decision-dependence"
DIRECTION_TRAINED_DISPLACEMENT = "trained-displacement"
REQUIRED_DIRECTIONS = frozenset(
    {
        DIRECTION_COSTLY_OTHER_REGARD,
        DIRECTION_DECISION_DEPENDENCE,
        DIRECTION_TRAINED_DISPLACEMENT,
    }
)
REQUIRED_CONSTRUCTS = frozenset({DIRECTION_COSTLY_OTHER_REGARD, DIRECTION_DECISION_DEPENDENCE})

GRADIENT_GATE_AUTOGRAD = "autograd"
REQUIRED_GRADIENT_GATES = frozenset({GRADIENT_GATE_AUTOGRAD, "kernel", "resume"})


@dataclass(frozen=True)
class BoundedCorpusPlan:
    """Exact fit/quality membership and token window for one bounded run."""

    profile: str
    fit_prompts: tuple[str, ...]
    quality_prompts: tuple[str, ...]
    fit_prompt_sha256: str
    quality_prompt_sha256: str
    fit_stimuli_sha256: str
    quality_stimuli_sha256: str
    fit_prompt_ids: tuple[str, ...]
    quality_prompt_ids: tuple[str, ...]
    fit_group_ids: tuple[str, ...]
    quality_group_ids: tuple[str, ...]
    max_seq_len: int
    corpus_max_tokens: int
    n_truncated: int

    @property
    def identity_sha256(self) -> str:
        """Digest every corpus/config field which changes what the fit or quality read sees."""
        return digest_strings([json.dumps(self.as_payload(), sort_keys=True)])

    def as_payload(self) -> dict[str, object]:
        """JSON-safe corpus identity, retaining membership without private prompt text."""
        payload = asdict(self)
        payload.pop("fit_prompts")
        payload.pop("quality_prompts")
        return payload


@dataclass(frozen=True)
class PreparedFitPrompts:
    """Exact private fit prompts and window shared by the gate producer and lens consumer."""

    stimuli: tuple[Stimulus, ...]
    rendered: dict[str, str]
    token_ids: dict[str, list[int]]
    prompts: tuple[str, ...]
    stimuli_sha256: str
    rendered_sha256: str
    max_seq_len: int

    def binding_payload(
        self,
        *,
        convention: str,
        enable_thinking: bool,
        tokenizer_content_identity: str,
    ) -> dict[str, object]:
        """Return the content/config binding persisted by the external gradient gate."""
        return {
            "fit_stimuli_sha256": self.stimuli_sha256,
            "fit_rendered_sha256": self.rendered_sha256,
            "stimulus_render": convention,
            "enable_thinking": enable_thinking,
            "tokenizer_content_sha256": tokenizer_content_identity,
            "n_fit_prompts": len(self.prompts),
        }


def prepare_fit_prompts(
    path: Path,
    tokenizer: PreTrainedTokenizerBase,
    *,
    convention: str,
    enable_thinking: bool,
    max_seq_len_ceiling: int,
) -> PreparedFitPrompts:
    """Load, render and tokenize the cooperation fit corpus through the canonical capture path."""
    fit_stimuli = tuple(load_stimuli(validate_private_stimulus_path(path)))
    rendered = render_stimuli(
        tokenizer,
        fit_stimuli,
        convention=convention,
        enable_thinking=enable_thinking,
    )
    token_ids = _tokenize(tokenizer, rendered)
    prompts = tuple(rendered[row.stimulus_id] for row in fit_stimuli)
    seq_plan = derive_max_seq_len(
        [len(token_ids[row.stimulus_id]) for row in fit_stimuli], ceiling=max_seq_len_ceiling
    )
    return PreparedFitPrompts(
        stimuli=fit_stimuli,
        rendered=rendered,
        token_ids=token_ids,
        prompts=prompts,
        stimuli_sha256=stimuli_digest(fit_stimuli),
        rendered_sha256=digest_strings(prompts),
        max_seq_len=seq_plan.max_seq_len,
    )


def _validate_complete_groups(stimuli: Sequence[Stimulus], *, role: str) -> tuple[str, ...]:
    """Require complete pairs and return their explicit scenario/template groups."""
    by_pair: dict[str, list[Stimulus]] = {}
    for stimulus in stimuli:
        by_pair.setdefault(stimulus.pair_id, []).append(stimulus)
    found_constructs = {stimulus.stimulus_set for stimulus in stimuli}
    if found_constructs != set(REQUIRED_CONSTRUCTS):
        raise ValueError(
            f"{role} corpus needs exactly constructs {sorted(REQUIRED_CONSTRUCTS)}, got "
            f"{sorted(found_constructs)}"
        )
    missing_prefixes = [
        stimulus.stimulus_id
        for stimulus in stimuli
        if stimulus.assistant_prefix is None or not stimulus.assistant_prefix.strip()
    ]
    if missing_prefixes:
        raise ValueError(
            f"{role} corpus has no short teacher-forced reasoning prefix for {missing_prefixes[:5]}"
        )
    malformed = {
        pair: [row.side for row in rows]
        for pair, rows in by_pair.items()
        if len(rows) != 2 or {row.side for row in rows} != {"A", "B"}  # noqa: PLR2004
    }
    if malformed:
        examples = dict(sorted(malformed.items())[:5])
        raise ValueError(
            f"{role} corpus has {len(malformed)} incomplete contrast groups {examples}; each "
            "pair_id needs exactly two distinct sides"
        )
    group_by_pair: dict[str, str] = {}
    for pair_id, members in by_pair.items():
        groups = {construct_group(member) for member in members}
        if len(groups) != 1:
            raise ValueError(f"{role} pair {pair_id!r} straddles scenario/template groups {groups}")
        group_by_pair[pair_id] = groups.pop()
    return tuple(sorted(set(group_by_pair.values())))


def validate_private_stimulus_path(path: Path) -> Path:
    """Require authored calibration text to stay under a gitignored private root."""
    resolved = path.resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in PRIVATE_STIMULUS_ROOTS):
        raise ValueError(
            f"lens stimulus text must live under one of {PRIVATE_STIMULUS_ROOTS}, got {path}"
        )
    return resolved


def validate_base_model_revision(base_model: str, base_revision: str | None) -> None:
    """Require a pinned hub revision and refuse meaningless revisions on local snapshots."""
    is_local_snapshot = Path(base_model).is_dir()
    if is_local_snapshot and base_revision is not None:
        raise ValueError("--base-revision must be omitted when --base-model is a local snapshot")
    if not is_local_snapshot and base_revision is None:
        raise ValueError("--base-revision is required when --base-model is a hub model ID")


def validate_construct_capture(
    root: Path, final_adapter: Path, *, final_arm: str, final_step: int
) -> dict[str, Any]:
    """Load the base/final construct capture, invoking its adapter-moved positive control."""
    ladder = load_ladder(root, require_capture_provenance=True)
    expected_labels = {f"{BASE_ARM}/step-{BASE_STEP}", f"{final_arm}/step-{final_step}"}
    labels = {cell.label for cell in ladder.cells}
    if labels != expected_labels:
        raise ValueError(
            f"construct capture must contain exactly base and final cells {sorted(expected_labels)}, "
            f"got {sorted(labels)}"
        )
    final_cell = ladder.cell(final_arm, final_step)
    adapter_sha256, _ = adapter_digests(final_adapter)
    if final_cell.adapter_weights_sha256 != adapter_sha256:
        raise ValueError(
            "construct capture final cell used different adapter weights from this lens run"
        )
    return {
        "root": str(root),
        "identity": ladder.identity.to_payload(),
        "cells": [cell.label for cell in ladder.cells],
        "adapter_positive_control": (
            "passed: load_ladder refused any adapted cell bit-identical to the base"
        ),
        "final_adapter_weights_sha256": adapter_sha256,
    }


def build_corpus_plan(  # noqa: C901, PLR0913 - explicit corpus contract checks
    fit_stimuli: Sequence[Stimulus],
    quality_stimuli: Sequence[Stimulus],
    *,
    rendered_fit: Mapping[str, str],
    rendered_quality: Mapping[str, str],
    fit_token_ids: Mapping[str, Sequence[int]],
    quality_token_ids: Mapping[str, Sequence[int]],
    profile: str,
    max_seq_len_ceiling: int,
    reserved_group_ids: set[str] | frozenset[str] = frozenset(),
    reserved_pair_ids: set[str] | frozenset[str] = frozenset(),
) -> BoundedCorpusPlan:
    """Validate the bounded corpus and refuse leakage or any cropped contrast-bearing prompt."""
    if profile not in PROFILE_FIT_PROMPTS:
        raise ValueError(f"unknown profile {profile!r}; expected {sorted(PROFILE_FIT_PROMPTS)}")
    expected_fit = PROFILE_FIT_PROMPTS[profile]
    if len(fit_stimuli) != expected_fit:
        raise ValueError(
            f"{profile} needs exactly {expected_fit} fit prompts, got {len(fit_stimuli)}; "
            "the ten- and fifty-prompt corpora are separate declared artifacts"
        )
    if len(quality_stimuli) != N_QUALITY_PROMPTS:
        raise ValueError(
            f"each state needs exactly {N_QUALITY_PROMPTS} quality prompts, got "
            f"{len(quality_stimuli)}"
        )
    fit_groups = _validate_complete_groups(fit_stimuli, role="fit")
    quality_groups = _validate_complete_groups(quality_stimuli, role="quality")
    corpus_pair_ids = {stimulus.pair_id for stimulus in [*fit_stimuli, *quality_stimuli]}
    pair_overlap = sorted(corpus_pair_ids & reserved_pair_ids)
    if pair_overlap:
        raise ValueError(
            "lens fit/quality pair ids overlap reserved training/evaluation ids "
            f"({pair_overlap[:5]})"
        )
    overlap = sorted(set(fit_groups) & set(quality_groups))
    if overlap:
        raise ValueError(
            f"fit and quality scenario groups overlap ({overlap[:5]}); held-out lens quality "
            "must use disjoint metadata.scenario_group values"
        )
    reserved_overlap = sorted((set(fit_groups) | set(quality_groups)) & reserved_group_ids)
    if reserved_overlap:
        raise ValueError(
            "lens fit/quality groups overlap reserved training/evaluation groups "
            f"({reserved_overlap[:5]}); calibration, quality, training and evaluation membership "
            "must be globally disjoint"
        )
    fit_ids = [row.stimulus_id for row in fit_stimuli]
    quality_ids = [row.stimulus_id for row in quality_stimuli]
    prompt_overlap = sorted(set(fit_ids) & set(quality_ids))
    if prompt_overlap:
        raise ValueError(f"fit and quality prompt ids overlap ({prompt_overlap[:5]})")
    for label, ids, render_map, token_map in (
        ("fit", fit_ids, rendered_fit, fit_token_ids),
        ("quality", quality_ids, rendered_quality, quality_token_ids),
    ):
        if set(ids) != set(render_map) or set(ids) != set(token_map):
            raise ValueError(
                f"{label} rendering/tokenization does not cover exactly its stimulus ids"
            )
    fit_prompts = tuple(rendered_fit[row_id] for row_id in fit_ids)
    quality_prompts = tuple(rendered_quality[row_id] for row_id in quality_ids)
    fit_token_lengths = [len(fit_token_ids[row_id]) for row_id in fit_ids]
    all_token_lengths = [
        *fit_token_lengths,
        *(len(quality_token_ids[row_id]) for row_id in quality_ids),
    ]
    seq_plan = derive_max_seq_len(fit_token_lengths, ceiling=max_seq_len_ceiling)
    lost_contrasts: list[str] = []
    for role, role_stimuli, role_token_ids in (
        ("fit", fit_stimuli, fit_token_ids),
        ("quality", quality_stimuli, quality_token_ids),
    ):
        divergences = pair_divergence_indices(
            role_stimuli, {key: list(value) for key, value in role_token_ids.items()}
        )
        lost_contrasts.extend(
            f"{role}:{pair_id}"
            for pair_id, index in divergences.items()
            if index >= seq_plan.max_seq_len
        )
    if lost_contrasts:
        raise ValueError(
            f"max_seq_len={seq_plan.max_seq_len} would crop contrast-bearing prompts: "
            f"the differing token is outside the window for {lost_contrasts[:5]}. Raise the "
            "ceiling; tail truncation is allowed only after each pair's contrast survives."
        )
    return BoundedCorpusPlan(
        profile=profile,
        fit_prompts=fit_prompts,
        quality_prompts=quality_prompts,
        fit_prompt_sha256=digest_strings(fit_prompts),
        quality_prompt_sha256=digest_strings(quality_prompts),
        fit_stimuli_sha256=stimuli_digest(fit_stimuli),
        quality_stimuli_sha256=stimuli_digest(quality_stimuli),
        fit_prompt_ids=tuple(fit_ids),
        quality_prompt_ids=tuple(quality_ids),
        fit_group_ids=fit_groups,
        quality_group_ids=quality_groups,
        max_seq_len=seq_plan.max_seq_len,
        corpus_max_tokens=max(all_token_lengths),
        n_truncated=sum(length > seq_plan.max_seq_len for length in all_token_lengths),
    )


@dataclass(frozen=True)
class LensRunIdentity:
    """Identity beside one resumable fp32 accumulation."""

    profile: str
    state: str
    lens_cache_key_sha256: str
    fit_prompt_sha256: str
    quality_prompt_sha256: str
    max_seq_len: int
    dim_batch: int
    checkpoint_every: int

    def as_payload(self) -> dict[str, object]:
        """JSON-safe exact identity written beside an accumulation."""
        return asdict(self)


def accumulation_identity_path(checkpoint_path: Path) -> Path:
    """Sidecar path paired with the reference fitter's accumulator."""
    return checkpoint_path.with_name(checkpoint_path.name + ACCUMULATION_IDENTITY_SUFFIX)


def prepare_accumulation_resume(checkpoint_path: Path, identity: LensRunIdentity) -> bool:
    """Return whether to resume, refusing every incomplete or incompatible checkpoint pair.

    The reference fitter stores no corpus/config identity inside its fp32 checkpoint.  This sidecar
    is therefore written before a fresh fit and compared byte-for-byte before ``resume=True`` is
    passed to it.  A checkpoint without a sidecar cannot be guessed into compatibility. A matching
    sidecar without a checkpoint is a fresh start (for example, the previous invocation hit the
    completed-lens cache or failed before jlens wrote its first accumulation).
    """
    sidecar = accumulation_identity_path(checkpoint_path)
    checkpoint_exists = checkpoint_path.is_file()
    sidecar_exists = sidecar.is_file()
    wanted = identity.as_payload()
    if checkpoint_exists and not sidecar_exists:
        raise ValueError(
            f"incomplete resumable accumulation at {checkpoint_path}: checkpoint_exists="
            f"{checkpoint_exists}, identity_exists={sidecar_exists}"
        )
    if sidecar_exists:
        found = cast("dict[str, object]", json.loads(sidecar.read_text()))
        if found != wanted:
            raise ValueError(
                f"accumulation identity mismatch at {checkpoint_path}; refusing to resume an "
                "fp32 sum from another corpus, model state, or fit configuration"
            )
        return checkpoint_exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(json.dumps(wanted, indent=2, sort_keys=True) + "\n")
    temporary.replace(sidecar)
    return False


def validate_gradient_gate(payload: dict[str, Any]) -> dict[str, Any]:
    """Require saved evidence that the real DeltaNet gradient and its sabotage both worked."""
    requested = set(cast("list[str]", payload.get("gates_requested", [])))
    missing = sorted(REQUIRED_GRADIENT_GATES - requested)
    if missing:
        raise ValueError(f"gradient gate report did not request required gates {missing}")
    gates = cast("dict[str, dict[str, Any]]", payload.get("gates", {}))
    absent = sorted(REQUIRED_GRADIENT_GATES - set(gates))
    if absent:
        raise ValueError(f"gradient gate report has no results for {absent}")
    autograd = gates[GRADIENT_GATE_AUTOGRAD]
    verdicts = cast("dict[str, object]", autograd.get("verdicts", {}))
    if verdicts.get("autograd_traverses_recurrence") is not True:
        raise ValueError(
            "gradient gate reports a zero or disconnected cross-position Jacobian path"
        )
    if verdicts.get("sabotage_reads_exactly_zero") is not True:
        raise ValueError("gradient gate's detach sabotage did not read exactly zero")
    failed = [name for name in REQUIRED_GRADIENT_GATES if gates[name].get("passed") is not True]
    if failed or payload.get("passed") is not True:
        raise ValueError(f"required Jacobian fit gates failed: {sorted(failed)}")
    return payload


def validate_gradient_gate_binding(  # noqa: PLR0913 - every runtime identity is load-bearing
    payload: dict[str, Any],
    *,
    base_weights_identity: str,
    tokenizer_content_identity: str,
    comparison_reference_identity: str,
    fit_stimuli_sha256: str,
    fit_rendered_sha256: str,
    stimulus_render: str,
    enable_thinking: bool,
    max_seq_len: int,
    dim_batch: int,
    kernels: Mapping[str, str],
) -> dict[str, Any]:
    """Bind external 9B gate evidence to this run's bytes, tokenizer, kernels and schedule."""
    validate_gradient_gate(payload)
    model = cast("dict[str, object]", payload.get("model", {}))
    if model.get("resolved_weights_identity") != base_weights_identity:
        raise ValueError(
            "gradient gate base weights identity differs from the model this lens run resolved"
        )
    token_gate = cast(
        "dict[str, object]", cast("dict[str, Any]", payload["gates"]).get("tokens", {})
    )
    if (
        token_gate.get("passed") is not True
        or token_gate.get("reference_identity") != comparison_reference_identity
    ):
        raise ValueError(
            "gradient gate tokenizer identity differs from the base tokenizer snapshot"
        )
    if model.get("tokenizer_content_sha256") != tokenizer_content_identity:
        raise ValueError(
            "gradient gate tokenizer content fingerprint differs from the loaded lens tokenizer"
        )
    expected_corpus_binding = {
        "fit_stimuli_sha256": fit_stimuli_sha256,
        "fit_rendered_sha256": fit_rendered_sha256,
        "stimulus_render": stimulus_render,
        "enable_thinking": enable_thinking,
        "tokenizer_content_sha256": tokenizer_content_identity,
    }
    corpus_binding = cast("dict[str, object]", payload.get("cooperation_fit_corpus", {}))
    stale_corpus_fields = [
        field
        for field, expected in expected_corpus_binding.items()
        if corpus_binding.get(field) != expected
    ]
    if stale_corpus_fields:
        raise ValueError(
            "gradient gate cooperation fit corpus/render binding differs from this lens run for "
            f"{stale_corpus_fields}"
        )
    if payload.get("max_seq_len") != max_seq_len:
        raise ValueError(
            f"gradient gate max_seq_len={payload.get('max_seq_len')} differs from fit "
            f"max_seq_len={max_seq_len}"
        )
    sweep = cast("dict[str, Any]", cast("dict[str, Any]", payload["gates"]).get("sweep", {}))
    choice = cast("dict[str, object]", sweep.get("choice", {}))
    if sweep.get("passed") is not True or choice.get("chosen") != dim_batch:
        raise ValueError(
            f"gradient gate selected dim_batch={choice.get('chosen')}, lens run asks for {dim_batch}"
        )
    if payload.get("deltanet_kernels_bound") != dict(kernels):
        raise ValueError("gradient gate kernel bindings differ from this lens process")
    jlens = cast("dict[str, object]", payload.get("jlens", {}))
    commit = jlens.get("commit")
    if not isinstance(commit, str) or not commit.startswith(JLENS_COMMIT):
        raise ValueError(
            f"gradient gate used jlens commit {commit!r}, not pinned commit {JLENS_COMMIT}"
        )
    return payload


def _quality_by_layer(quality: Mapping[str, Any]) -> dict[int, dict[str, float]]:
    """Join Jacobian and logit-lens residuals by layer for direction readouts."""
    jacobian = {
        int(row["layer"]): float(row["relative_residual"])
        for row in quality.get("jacobian", {}).get("per_layer", [])
    }
    logit = {
        int(row["layer"]): float(row["relative_residual"])
        for row in quality.get("logit_lens_baseline", {}).get("per_layer", [])
    }
    if not jacobian or jacobian.keys() != logit.keys():
        raise ValueError(
            "direction decoding needs matching per-layer Jacobian and logit-lens quality"
        )
    return {
        layer: {
            "jacobian_relative_residual": jacobian[layer],
            "logit_lens_relative_residual": logit[layer],
        }
        for layer in sorted(jacobian)
    }


def decode_directions(  # noqa: PLR0913 - one read is lens, model, tokenizer, paths and knobs
    lens: object,
    model: object,
    tokenizer: object,
    direction_paths: Mapping[str, Path],
    *,
    quality: Mapping[str, Any],
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    """Decode all required axes and matched-norm controls, with quality beside every layer."""
    names = set(direction_paths)
    if names != set(REQUIRED_DIRECTIONS):
        raise ValueError(
            f"direction map must contain exactly {sorted(REQUIRED_DIRECTIONS)}, got {sorted(names)}"
        )
    layer_quality = _quality_by_layer(quality)

    def decode_id(token_id: int) -> str:
        return cast("str", cast("Any", tokenizer).convert_ids_to_tokens(token_id))

    output: dict[str, Any] = {}
    for name, path in sorted(direction_paths.items()):
        directions = cast("dict[int, torch.Tensor]", torch.load(path, weights_only=True))
        if not directions:
            raise ValueError(f"{path} contains no per-layer directions")
        layers: dict[str, Any] = {}
        for layer, saved_direction in sorted(directions.items()):
            if layer not in layer_quality:
                raise ValueError(
                    f"{name} layer {layer} has no fit-quality row; every interpretation must carry "
                    "Jacobian and logit-lens quality beside it"
                )
            direction = saved_direction.float()
            direction_norm = float(direction.norm())
            if direction_norm == 0.0:
                raise ValueError(f"{name} layer {layer} is a zero direction")
            generator = torch.Generator().manual_seed(seed + zlib.crc32(f"{name}:{layer}".encode()))
            random_direction = matched_norm_random_direction(direction, generator)

            def rows(vector: torch.Tensor, *, source_layer: int = layer) -> list[dict[str, object]]:
                return [
                    {"token": item.token, "logit": item.logit}
                    for item in transport_and_decode(
                        cast("Any", lens),
                        cast("Any", model),
                        vector,
                        source_layer,
                        id_to_token=decode_id,
                        k=top_k,
                    )
                ]

            layers[str(layer)] = {
                "quality": layer_quality[layer],
                "direction_norm": direction_norm,
                "random_norm": float(random_direction.norm()),
                "real": rows(direction),
                "matched_norm_random": rows(random_direction),
            }
        output[name] = {
            "path": str(path),
            "sha256": sha256_of_file(path),
            "layers": layers,
        }
    return output


def shared_coordinate_payload(
    quality: Mapping[str, Any], direction_readouts: Mapping[str, Any]
) -> dict[str, Any]:
    """Label the shared-base-lens read as an approximation with its fidelity attached."""
    return {
        "readout_kind": "shared-base-coordinate-sensitivity",
        "approximation": True,
        "interpretation": (
            "The base lens is applied to final-checkpoint states to hold readout coordinates "
            "fixed. Its per-layer cross-checkpoint reconstruction fidelity is reported; this "
            "comparison is coordinate sensitivity, not a model-specific final read."
        ),
        "lens_state": STATE_BASE,
        "applied_to_state": STATE_FINAL,
        "cross_checkpoint_fidelity": dict(quality),
        "direction_readouts": dict(direction_readouts),
    }


def model_specific_metadata(state: str) -> dict[str, object]:
    """Label a lens fitted and applied on the same model state."""
    if state not in STATES:
        raise ValueError(f"model-specific lens state must be one of {list(STATES)}, got {state!r}")
    return {
        "readout_kind": "model-specific",
        "approximation": False,
        "lens_state": state,
        "applied_to_state": state,
    }


def load_direction_manifest(path: Path) -> dict[str, dict[str, Path]]:
    """Load exact base/final paths for the two construct axes and trained displacement."""
    payload = cast("dict[str, dict[str, str]]", json.loads(path.read_text()))
    if set(payload) != set(STATES):
        raise ValueError(f"direction manifest needs exactly states {list(STATES)}")
    result: dict[str, dict[str, Path]] = {}
    for state in STATES:
        if set(payload[state]) != set(REQUIRED_DIRECTIONS):
            raise ValueError(
                f"direction manifest state {state!r} needs exactly {sorted(REQUIRED_DIRECTIONS)}"
            )
        result[state] = {name: Path(value) for name, value in payload[state].items()}
        missing = [str(value) for value in result[state].values() if not value.is_file()]
        if missing:
            raise FileNotFoundError(
                f"direction manifest state {state!r} has missing files {missing}"
            )
    return result


@dataclass(frozen=True)
class ReservedIdentities:
    """Training/evaluation identities excluded from both lens corpora."""

    group_ids: frozenset[str]
    pair_ids: frozenset[str]


def load_reserved_identities(path: Path) -> ReservedIdentities:
    """Read reserved group and pair IDs; a legacy JSON list denotes group IDs."""
    payload = cast("object", json.loads(path.read_text()))
    if isinstance(payload, list):
        groups, pairs = payload, []
    elif isinstance(payload, dict):
        unknown = set(payload) - {"group_ids", "pair_ids"}
        if unknown:
            raise ValueError(f"{path} has unknown reserved identity keys {sorted(unknown)}")
        groups = cast("list[object]", payload.get("group_ids", []))
        pairs = cast("list[object]", payload.get("pair_ids", []))
    else:
        raise TypeError(f"{path} must contain a JSON list or mapping of lists")
    identities = ReservedIdentities(
        group_ids=frozenset(str(value) for value in groups),
        pair_ids=frozenset(str(value) for value in pairs),
    )
    if not identities.group_ids and not identities.pair_ids:
        raise ValueError(f"{path} contains no reserved training/evaluation identities")
    return identities


def _tokenize(tokenizer: object, rendered: Mapping[str, str]) -> dict[str, list[int]]:
    """Tokenize rendered prompts one at a time, preserving their id association."""
    return {
        row_id: cast("list[int]", tokenizer(text, add_special_tokens=False)["input_ids"])  # pyright: ignore[reportCallIssue]
        for row_id, text in rendered.items()
    }


def _fit_state(  # noqa: PLR0913 - a complete durable state read has several explicit identities
    *,
    state: str,
    cell: LadderCell,
    target: object,
    jl: object,
    corpus: BoundedCorpusPlan,
    out_dir: Path,
    cache: LensCache,
    cache_key: LensCacheKey,
    dim_batch: int,
    checkpoint_every: int,
    eval_positions: int,
    direction_paths: Mapping[str, Path],
    top_k: int,
    seed: int,
) -> tuple[object, dict[str, Any]]:
    """Fit/load, score and decode one model-specific state lens."""
    state_dir = out_dir / state
    accumulator_path = state_dir / ACCUMULATION_FILENAME
    resume_identity = LensRunIdentity(
        profile=corpus.profile,
        state=state,
        lens_cache_key_sha256=cache_key.sha256,
        fit_prompt_sha256=corpus.fit_prompt_sha256,
        quality_prompt_sha256=corpus.quality_prompt_sha256,
        max_seq_len=corpus.max_seq_len,
        dim_batch=dim_batch,
        checkpoint_every=checkpoint_every,
    )
    resume = prepare_accumulation_resume(accumulator_path, resume_identity)
    model = target.model  # pyright: ignore[reportAttributeAccessIssue]
    tokenizer = target.tokenizer  # pyright: ignore[reportAttributeAccessIssue]
    config = JacobianConfig(
        model_id=target.label,  # pyright: ignore[reportAttributeAccessIssue]
        source="fit_own",
        max_fit_prompts=len(corpus.fit_prompts),
        dim_batch=dim_batch,
        max_seq_len=corpus.max_seq_len,
        checkpoint_path=accumulator_path,
        checkpoint_every=checkpoint_every,
        resume=resume,
        top_k=top_k,
    )
    acquisition = acquire_lens(
        config,
        model,
        corpus.fit_prompts,
        cast("Any", jl),
        lens_path=state_dir / LENS_FILENAME,
        cache=cache,
        key=cache_key,
    )
    quality_report = evaluate_reconstruction(
        acquisition.lens,
        model,
        corpus.quality_prompts,
        max_seq_len=corpus.max_seq_len,
        max_positions=eval_positions,
    )
    quality = fit_quality_payload(quality_report)
    if quality.get("available") is not True:
        raise RuntimeError(f"{state} lens quality is unavailable: {quality.get('reason')}")
    entry = {
        **model_specific_metadata(state),
        "state": state,
        "arm": cell.arm,
        "step": cell.step,
        "model": target.label,  # pyright: ignore[reportAttributeAccessIssue]
        "lens_model": LENS_MODEL_UNMERGED,
        "lens_path": str(state_dir / LENS_FILENAME),
        "lens_cache_key": cache_key.as_payload(),
        "lens_acquisition": acquisition.as_payload(),
        "accumulation_identity": resume_identity.as_payload(),
        "fit_quality": quality,
        "direction_readouts": decode_directions(
            acquisition.lens,
            model,
            tokenizer,
            direction_paths,
            quality=quality,
            top_k=top_k,
            seed=seed,
        ),
    }
    return acquisition.lens, entry


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the base/final lens pair and durable shared-base coordinate check."""
    validate_base_model_revision(
        cast("str", args.base_model), cast("str | None", args.base_revision)
    )
    gradient_path = cast("Path", args.gradient_gate_report)
    gradient_payload = validate_gradient_gate(
        cast("dict[str, Any]", json.loads(gradient_path.read_text()))
    )
    quality_path = validate_private_stimulus_path(cast("Path", args.quality_stimuli))
    quality_stimuli = load_stimuli(quality_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, revision=args.base_revision, trust_remote_code=True
    )
    prepared_fit = prepare_fit_prompts(
        cast("Path", args.fit_stimuli),
        tokenizer,
        convention=cast("str", args.stimulus_render),
        enable_thinking=not cast("bool", args.no_thinking),
        max_seq_len_ceiling=cast("int", args.max_seq_len_ceiling),
    )
    rendered_quality = render_stimuli(
        tokenizer,
        quality_stimuli,
        convention=cast("str", args.stimulus_render),
        enable_thinking=not cast("bool", args.no_thinking),
    )
    reserved = load_reserved_identities(cast("Path", args.reserved_group_ids))
    corpus = build_corpus_plan(
        prepared_fit.stimuli,
        quality_stimuli,
        rendered_fit=prepared_fit.rendered,
        rendered_quality=rendered_quality,
        fit_token_ids=prepared_fit.token_ids,
        quality_token_ids=_tokenize(tokenizer, rendered_quality),
        profile=args.profile,
        max_seq_len_ceiling=args.max_seq_len_ceiling,
        reserved_group_ids=reserved.group_ids,
        reserved_pair_ids=reserved.pair_ids,
    )
    directions = load_direction_manifest(cast("Path", args.direction_manifest))
    final_adapter = cast("Path", args.final_adapter)
    capture_validation = validate_construct_capture(
        cast("Path", args.construct_capture_root),
        final_adapter,
        final_arm=cast("str", args.final_arm),
        final_step=cast("int", args.final_step),
    )
    cells = [
        LadderCell(arm=BASE_ARM, step=BASE_STEP, adapter_dir=None),
        LadderCell(arm=args.final_arm, step=args.final_step, adapter_dir=final_adapter),
    ]
    assert_one_adapter_config([final_adapter])
    jl = _require_jlens()
    bridge = bridge_and_check_decode_kernel()
    kernels = bound_deltanet_kernels()
    base_identity = resolve_weights_identity(args.base_model, revision=args.base_revision)
    # lens_schedule_gates.load_tokenizer_encoders records resolve_weights_identity verbatim.
    comparison_reference_identity = f"hf:{DEFAULT_BASE_REVISION}"
    tokenizer_content_identity = tokenizer_content_sha256(tokenizer)
    validate_gradient_gate_binding(
        gradient_payload,
        base_weights_identity=base_identity,
        tokenizer_content_identity=tokenizer_content_identity,
        comparison_reference_identity=comparison_reference_identity,
        fit_stimuli_sha256=corpus.fit_stimuli_sha256,
        fit_rendered_sha256=corpus.fit_prompt_sha256,
        stimulus_render=cast("str", args.stimulus_render),
        enable_thinking=not cast("bool", args.no_thinking),
        max_seq_len=corpus.max_seq_len,
        dim_batch=args.dim_batch,
        kernels=kernels,
    )
    models = LadderLensModels(
        base_model=args.base_model,
        dtype=torch.bfloat16,
        jl=jl,
        lens_model=LENS_MODEL_UNMERGED,
        merge_root=args.out_dir / "unused-merged",
        base_weights_identity=base_identity,
        base_revision=args.base_revision,
    )
    cache = LensCache(root=str(args.lens_cache or (args.out_dir / "cache")))
    report: dict[str, Any] = {
        "git_sha": git_sha(),
        "profile": corpus.profile,
        "base_model": args.base_model,
        "base_revision": args.base_revision,
        "base_weights_identity": base_identity,
        "tokenizer_content_sha256": tokenizer_content_identity,
        "token_comparison_reference_identity": comparison_reference_identity,
        "final_adapter": str(final_adapter),
        "final_arm": args.final_arm,
        "final_step": args.final_step,
        "construct_capture": capture_validation,
        "corpus": corpus.as_payload(),
        "corpus_identity_sha256": corpus.identity_sha256,
        "gradient_gate_report": str(gradient_path),
        "gradient_gate_sha256": sha256_of_file(gradient_path),
        "deltanet_kernel_bridge": bridge,
        "deltanet_kernels_bound": kernels,
        DELTANET_KERNEL_FIELD: prefill_deltanet_kernels(kernels),
        "states": {},
        "shared_base_lens_coordinate_sensitivity": None,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / REPORT_FILENAME
    base_lens_path = args.out_dir / STATE_BASE / LENS_FILENAME
    for state, cell in zip(STATES, cells, strict=True):
        target = models.target_for(cell)
        cache_key = lens_cache_key_for_cell(
            cell,
            base_model=args.base_model,
            base_weights_identity=base_identity,
            merge_dtype=torch.bfloat16,
            fit_prompts=corpus.fit_prompts,
            max_seq_len=corpus.max_seq_len,
            skip_first=fit_skip_first(jl),
            dim_batch=args.dim_batch,
            load_path=JLENS_UNMERGED_LOAD_PATH,
        )
        lens, entry = _fit_state(
            state=state,
            cell=cell,
            target=target,
            jl=jl,
            corpus=corpus,
            out_dir=args.out_dir,
            cache=cache,
            cache_key=cache_key,
            dim_batch=args.dim_batch,
            checkpoint_every=args.checkpoint_every,
            eval_positions=args.eval_positions,
            direction_paths=directions[state],
            top_k=args.top_k,
            seed=args.seed,
        )
        report["states"][state] = entry
        if state == STATE_BASE:
            del lens
            gc.collect()
            torch.cuda.empty_cache()
        else:
            # The two 9B lenses can each be several GiB. Release the final lens before reloading the
            # saved base lens for the shared-coordinate read, so both never occupy memory together.
            del lens
            gc.collect()
            torch.cuda.empty_cache()
            base_lens = jl.JacobianLens.load(str(base_lens_path))
            shared_quality_report = evaluate_reconstruction(
                base_lens,
                target.model,
                corpus.quality_prompts,
                max_seq_len=corpus.max_seq_len,
                max_positions=args.eval_positions,
            )
            shared_quality = fit_quality_payload(shared_quality_report)
            if shared_quality.get("available") is not True:
                raise RuntimeError("shared base-lens cross-checkpoint fidelity is unavailable")
            shared_readouts = decode_directions(
                base_lens,
                target.model,
                target.tokenizer,
                directions[STATE_FINAL],
                quality=shared_quality,
                top_k=args.top_k,
                seed=args.seed,
            )
            report["shared_base_lens_coordinate_sensitivity"] = shared_coordinate_payload(
                shared_quality, shared_readouts
            )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded base/final lens CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-stimuli", type=Path, required=True)
    parser.add_argument("--quality-stimuli", type=Path, required=True)
    parser.add_argument("--direction-manifest", type=Path, required=True)
    parser.add_argument(
        "--construct-capture-root",
        type=Path,
        required=True,
        help="Base/final construct capture; loading it runs the adapter-moved positive control.",
    )
    parser.add_argument("--gradient-gate-report", type=Path, required=True)
    parser.add_argument(
        "--reserved-group-ids",
        type=Path,
        required=True,
        help="JSON {group_ids, pair_ids} covering training, behavior evaluation and construct "
        "direction fit/held-out identities excluded from the separate lens corpora.",
    )
    parser.add_argument("--final-adapter", type=Path, required=True)
    parser.add_argument("--final-arm", required=True)
    parser.add_argument("--final-step", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_FIT_PROMPTS), required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--base-revision",
        default=None,
        help="Exact hub commit used for base weights/tokenizer; omit only for a local snapshot.",
    )
    parser.add_argument(
        "--stimulus-render",
        choices=("templated_here", "verbatim"),
        default="templated_here",
    )
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--max-seq-len-ceiling", type=int, default=DEFAULT_MAX_SEQ_LEN_CEILING)
    parser.add_argument("--dim-batch", type=int, default=JacobianConfig.dim_batch)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--eval-positions", type=int, default=DEFAULT_EVAL_POSITIONS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lens-cache", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded lens command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
