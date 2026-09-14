"""Capture residual-stream activations at every checkpoint of a games arm, from un-merged adapters.

One GPU pass, cached tensors, and every later interp read is arithmetic. This is the driver for that
pass: it walks a ladder of `(arm, checkpoint)` cells, applies each checkpoint's LoRA adapter at
runtime, runs one forward pass per stimulus, and writes the cells `games.interp_cells` defines.

**Runtime adapters, not merged exports.** A merge rounds `W + BA` back into bf16 and realizes only
about 64% of the trained delta (measured 2026-08-20, 38-79% per module). A before/after read whose
whole content is the size and direction of a change cannot afford a systematic two-thirds
attenuation, so the adapter is applied at runtime instead -- `Wx + B(Ax)` at full precision, through
the same load path training used (`games.lora.load_adapter_base` and `attach_adapter`). PEFT injects
into the model tree in place, so the capture drives the base model object it built and the adapter
applies; `assert_adapter_moved_activations` in the loader is what proves that end to end rather than
by argument.

**The render convention is chosen, checked, and recorded.** A corpus row can be either an untemplated
stem with a separate `assistant_prefix` (`templated_here`, which is what `games.interp_stimuli` emits:
this driver applies the chat template and appends the prefix, so the teacher-forced reasoning lands
inside the assistant turn's opening `<think>`) or the fully rendered string (`verbatim`, nothing added).
Getting it backwards is silent both ways, and one direction was measured on this very corpus on
2026-08-20: a pre-rendered row templated a second time nests the reasoning inside a *user* turn, the
assistant turn then ends at an empty `<think>`, and `last` pooling reads a position identical for both
sides of every pair -- 392 rows, every direction at noise, nothing raising.
`assert_render_convention` derives the template's own leading segment from the tokenizer and refuses
both mismatches: a `templated_here` corpus whose text already carries that segment, and a `verbatim`
corpus whose text does not. The convention is an identity field, and so is a digest of the strings the
model actually read, so a template that changes underneath a half-finished ladder is refusable too.

**Truncation never happens here, which is what makes one failure impossible rather than guarded.**
These sets are matched-stem, so the first token where a pair's two sides differ is late -- index
326-415 of a 368-451 token string, measured on this corpus. Any window shorter than that keeps only
the shared prefix, both sides become byte-identical, and every direction is exactly zero. So
`--max-prompt-tokens` is a refusal rather than a clamp: a corpus that does not fit stops the run. The
budget is also floored at `games.interp_stimuli.MIN_CAPTURE_SEQ_LEN`, so a window too short to hold
this corpus is refused against the number the operator typed rather than reported one step later as a
corpus that overruns its budget, which reads as the stimuli's fault.

**Batch size 1 is the default and is load-bearing.** Batch-1 forwards on this architecture are
bit-reproducible while a batch of 4 differs by 1.6-3.1% relative L2 at mid and late layers -- the
same order as the trained delta. Batching is available for a smoke, and is recorded in the cell
identity so a mixed-batching comparison is refused rather than read.

The forward pass runs the transformer trunk, not the full causal LM
(`reward_hacking.interp.directions.capture_pooled_activations_multi`), so the `[batch, seq, 248320]`
LM-head projection is never materialized -- at this vocabulary it OOMs a 24 GiB card on its own. Every
requested pooling comes off ONE forward, pooled inside the hook, so peak memory does not scale with
sequence length and a second pooling costs arithmetic rather than another pass over the corpus.

**Every cell records which Gated DeltaNet kernels its forwards dispatched** (`deltanet_kernel`). This
capture is forward-only: every forward is a prefill with no prior cache state, which transformers
routes through `chunk_gated_delta_rule` and `causal_conv1d_fn` and never through the per-token decode
pair the fla bridge re-binds. So the cell field carries those two
(`games.deltanet_kernels.prefill_deltanet_kernels`), which the bridge leaves unchanged, and a cell
captured on either side of the bridge names the same kernels because its tensors are the same. The
bridge is applied anyway, before the model loads, so a capture run in the same process as a
generating leg does not decide that leg's kernel by import order; the ladder manifest carries the
bridge report and the full four-kernel binding as process provenance, and nothing else -- a resume
skips cells already on disk, so a manifest-level `deltanet_kernel` would speak for cells this process
never ran, and a pooling read has to take each cell's own.

Resumable at cell granularity because it runs on spot: a cell whose manifest is already on disk is
skipped, and the manifest is written after the tensors so an interrupted cell is not mistaken for a
finished one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
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
from games.evals import SECTION_GAME_BEHAVIOR
from games.interp_cells import (
    BASE_ARM,
    BASE_STEP,
    CELL_MANIFEST_FILENAME,
    COMPUTE_DTYPES,
    LAYER_CONVENTION_POST_BLOCK,
    STIMULUS_RENDER_TEMPLATED,
    STIMULUS_RENDER_VERBATIM,
    STIMULUS_RENDERS,
    STORE_DTYPES,
    CellIdentity,
    Stimulus,
    construct_split_payload,
    digest_of_strings,
    group_by_set,
    load_private_construct_stimuli,
    load_stimuli,
    row_index_for,
    sha256_of_file,
    short_digest,
    step_dir,
    stimuli_digest,
    write_cell,
    write_ladder_manifest,
    write_natural_prefix_capture,
)
from games.interp_stimuli import MIN_CAPTURE_SEQ_LEN
from games.lora import (
    adapter_config_identity,
    assert_one_adapter_config,
    attach_adapter,
    checkpoint_step,
    load_adapter_base,
)
from games.preflight import derive_prefilled_think, resolve_chat_template_kwargs
from games.provenance import git_sha
from games.tokenizer_identity import tokenizer_content_sha256 as tokenizer_identity
from reward_hacking.interp.directions import (
    POOLERS,
    capture_pooled_activations_multi,
    capture_positionwise_activations,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from peft import PeftModel
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger("games.interp_capture")

ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_PROMPT_TOKENS = 4096
DEFAULT_POOLINGS = ("mean", "last")
PRIVATE_SELECTION_MANIFEST_ROOTS = (Path("docs/scratch"), Path("artifacts"))
MIN_NATURAL_REASONING_TOKENS = 3
NATURAL_STATES = ("base", "final")
NATURAL_RECORD_FIELDS = frozenset(
    {"request_id", "rollout_id", "scenario_group", "prompt", "pre_action_prefix"}
)

# Substituted into the chat template to find where a user turn's content begins. Not natural text.
RENDER_PROBE_SENTINEL = "__GAMES_INTERP_RENDER_PROBE__"

# How many offending ids an error quotes; the counts beside them carry the magnitude.
ERROR_EXAMPLE_COUNT = 5


@dataclass(frozen=True)
class LadderCell:
    """One unit of work: a model state to capture every stimulus set under."""

    arm: str
    step: int
    adapter_dir: Path | None

    @property
    def label(self) -> str:
        """How a cell names itself in a log line."""
        return f"{self.arm}/step-{self.step}"


@dataclass(frozen=True)
class ContrastBoundary:
    """The first token position where the two sides of one rendered pair differ."""

    pair_id: str
    first_differing_position: int
    left_length: int
    right_length: int


@dataclass(frozen=True)
class NaturalPrefixSelection:
    """A small, explicit set of positions retained from a natural rollout prefix."""

    stimulus_id: str
    positions: tuple[int, ...]
    selection_rule: str
    request_id: str | None = None
    rollout_id: str | None = None
    scenario_group: str | None = None
    natural_state: str | None = None


def kernel_identity(binding: dict[str, str]) -> str:
    """Fingerprint the prefill kernel binding used by a forward-only capture."""
    if not binding:
        raise ValueError("kernel binding is empty; a capture must record which kernel it used")
    return json.dumps(binding, sort_keys=True, separators=(",", ":"))


def _private_selection_path(path: Path) -> Path:
    """Resolve a natural-prefix artifact only when its location is private."""
    resolved = path.resolve()
    under_known_private_root = any(
        resolved.is_relative_to(root.resolve()) for root in PRIVATE_SELECTION_MANIFEST_ROOTS
    )
    if not under_known_private_root:
        ignored = subprocess.run(  # noqa: S603 - fixed git command; path is one argv value
            ["git", "check-ignore", "--no-index", "--quiet", str(resolved)],  # noqa: S607 - trusted literal command
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        under_known_private_root = ignored.returncode == 0
    if not under_known_private_root:
        raise ValueError(
            f"natural selection manifest must live under docs/scratch or artifacts, or be "
            f"gitignored, got {path}"
        )
    return resolved


def _validate_natural_rollout_ids(path: Path, payload: dict[str, Any]) -> None:
    """Validate the state-specific rollout identities in a selection manifest."""
    rollout_ids_by_state = payload["rollout_ids_by_state"]
    if (
        not isinstance(rollout_ids_by_state, dict)
        or set(rollout_ids_by_state) != set(NATURAL_STATES)
        or not all(
            isinstance(rollout_id, str) and rollout_id.strip()
            for rollout_id in rollout_ids_by_state.values()
        )
    ):
        raise ValueError(
            f"{path} rollout_ids_by_state must map exactly {list(NATURAL_STATES)} to non-empty strings"
        )
    if len(set(rollout_ids_by_state.values())) != len(rollout_ids_by_state):
        raise ValueError(f"{path} rollout_ids_by_state repeats an identity")


def _validate_natural_selection_payload(path: Path, payload: dict[str, Any]) -> list[int]:
    """Validate the manifest fields that freeze natural rollout selection."""
    required = {
        "version",
        "rollout_ids_by_state",
        "request_ids",
        "scenario_groups",
        "selection_rule",
        "layers",
    }
    if set(payload) != required:
        raise ValueError(f"{path} must contain exactly {sorted(required)}, got {sorted(payload)}")
    if payload["version"] != 1:
        raise ValueError(
            f"{path} has unsupported selection manifest version {payload['version']!r}"
        )
    _validate_natural_rollout_ids(path, payload)
    for field_name in ("request_ids", "scenario_groups"):
        values = payload[field_name]
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value.strip() for value in values)
        ):
            raise ValueError(f"{path} field {field_name!r} must be a non-empty list of strings")
        if len(set(values)) != len(values):
            raise ValueError(f"{path} field {field_name!r} repeats an identity")
    if not isinstance(payload["selection_rule"], str) or not payload["selection_rule"].strip():
        raise ValueError(f"{path} selection_rule must be a non-empty string")
    layers = payload["layers"]
    if (
        not isinstance(layers, list)
        or not layers
        or not all(
            isinstance(layer, int) and not isinstance(layer, bool) and layer >= 0
            for layer in layers
        )
    ):
        raise ValueError(f"{path} layers must be a non-empty list of non-negative integers")
    if len(set(layers)) != len(layers):
        raise ValueError(f"{path} layers repeats a decoder layer")
    return list(layers)


def load_natural_selection_manifest(path: Path) -> dict[str, Any]:
    """Load the predeclared natural-prefix source, rule and layer subset.

    The manifest is private because its source rollout/group identifiers can identify a future
    evaluation corpus. It is read before capture so positions cannot be selected after inspecting
    the model outputs.
    """
    resolved = _private_selection_path(path)
    raw_payload = json.loads(resolved.read_text())
    if not isinstance(raw_payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    payload = cast("dict[str, Any]", raw_payload)
    layers = _validate_natural_selection_payload(path, payload)
    return {
        **payload,
        "path": str(resolved),
        "sha256": sha256_of_file(resolved),
        "layers": layers,
    }


def load_capture_reservations(path: Path | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load globally reserved scenario and pair IDs for the cooperation construct corpus."""
    if path is None:
        return (), ()
    payload = cast("object", json.loads(path.read_text()))
    if isinstance(payload, list):
        groups, pairs = payload, []
    elif isinstance(payload, dict):
        unknown = set(payload) - {"group_ids", "pair_ids"}
        if unknown:
            raise ValueError(f"{path} has unknown reservation keys {sorted(unknown)}")
        groups, pairs = payload.get("group_ids", []), payload.get("pair_ids", [])
    else:
        raise TypeError(f"{path} must be a JSON list or {{group_ids, pair_ids}} object")
    if not isinstance(groups, list) or not isinstance(pairs, list):
        raise TypeError(f"{path} reservation group_ids and pair_ids must be JSON lists")
    group_ids = tuple(str(value) for value in groups)
    pair_ids = tuple(str(value) for value in pairs)
    if len(set(group_ids)) != len(group_ids) or len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f"{path} repeats a reserved group or pair identity")
    return group_ids, pair_ids


def _index_natural_behavior_records(
    records: Sequence[Mapping[str, Any]], expected_request_ids: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    """Index selected behavior records while ignoring unrelated trace sections."""
    expected = set(expected_request_ids)
    found: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if record.get("record", SECTION_GAME_BEHAVIOR) != SECTION_GAME_BEHAVIOR:
            continue
        missing = sorted(NATURAL_RECORD_FIELDS - set(record))
        if missing:
            raise ValueError(f"natural behavior record is missing fields {missing}")
        request_id = str(record["request_id"])
        if request_id not in expected:
            continue
        if request_id in found:
            raise ValueError(f"natural behavior records repeat request_id {request_id!r}")
        found[request_id] = record
    missing_requests = sorted(expected - set(found))
    if missing_requests:
        raise ValueError(
            f"natural selection manifest requests were absent from retained records: {missing_requests}"
        )
    return found


def _natural_stimulus_from_record(
    request_id: str,
    record: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    natural_state: str,
) -> Stimulus:
    """Validate selected record metadata and construct its model-facing stimulus."""
    rollout_id = str(record["rollout_id"])
    scenario_group = str(record["scenario_group"])
    expected_rollout_id = selection_manifest["rollout_ids_by_state"][natural_state]
    if rollout_id != expected_rollout_id:
        raise ValueError(
            f"natural record {request_id!r} has rollout_id {rollout_id!r}; "
            f"state {natural_state!r} requires {expected_rollout_id!r}"
        )
    if scenario_group not in selection_manifest["scenario_groups"]:
        raise ValueError(f"natural record {request_id!r} has an unregistered scenario_group")
    prefix = record["pre_action_prefix"]
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError(f"natural record {request_id!r} has no non-empty pre_action_prefix")
    prompt = record["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"natural record {request_id!r} has no non-empty prompt")
    return Stimulus(
        stimulus_id=f"natural-prefix--{request_id}",
        stimulus_set="natural-prefix",
        side="N",
        pair_id=request_id,
        text=prompt,
        assistant_prefix=prefix,
        metadata={
            "rollout_id": rollout_id,
            "natural_state": natural_state,
            "request_id": request_id,
            "scenario_group": scenario_group,
            "measurement_boundary": "pre_action",
            "action_commitment_present": False,
            "selection_rule": str(selection_manifest["selection_rule"]),
        },
    )


def _add_natural_selection_positions(
    stimulus: Stimulus,
    tokenizer: PreTrainedTokenizerBase,
    *,
    convention: str,
    enable_thinking: bool,
) -> Stimulus:
    """Render one selected prefix and attach prompt-boundary positions."""
    rendered = render_stimuli(
        tokenizer,
        [stimulus],
        convention=convention,
        enable_thinking=enable_thinking,
    )
    prompt_end = render_stimuli(
        tokenizer,
        [stimulus],
        convention=convention,
        enable_thinking=enable_thinking,
        include_assistant_prefix=False,
    )
    prompt_end_count = len(_token_ids(tokenizer, prompt_end[stimulus.stimulus_id]))
    full_count = len(_token_ids(tokenizer, rendered[stimulus.stimulus_id]))
    reasoning_count = full_count - prompt_end_count
    if prompt_end_count < 1 or reasoning_count < MIN_NATURAL_REASONING_TOKENS:
        request_id = str(stimulus.metadata["request_id"])
        raise ValueError(
            f"natural record {request_id!r} needs at least three pre-action reasoning tokens "
            f"after its prompt boundary, got {reasoning_count}"
        )
    selected_positions = [
        prompt_end_count - 1,
        prompt_end_count + reasoning_count // 2,
        full_count - 1,
    ]
    return replace(
        stimulus,
        metadata={
            **stimulus.metadata,
            "selected_positions": selected_positions,
            "prompt_end_token_position": prompt_end_count - 1,
        },
    )


def build_natural_prefix_stimuli(  # noqa: PLR0913 - explicit tokenizer and rendering controls
    records: Sequence[Mapping[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    selection_manifest: Mapping[str, Any],
    *,
    natural_state: str,
    convention: str,
    enable_thinking: bool,
) -> list[Stimulus]:
    """Extract pre-action prefixes for the exact predeclared behavior requests.

    Retained behavior records must carry ``request_id``, ``rollout_id``, ``scenario_group``,
    ``prompt`` and ``pre_action_prefix``. The latter is produced by the behavior parser while its
    action boundary is known; inferring a boundary by searching arbitrary completion prose would
    make the natural cross-check depend on a post-hoc parser guess.
    """
    if natural_state not in NATURAL_STATES:
        raise ValueError(
            f"natural_state must be one of {list(NATURAL_STATES)}, got {natural_state!r}"
        )
    expected_requests = tuple(str(value) for value in selection_manifest["request_ids"])
    if len(set(expected_requests)) != len(expected_requests):
        raise ValueError("natural selection manifest repeats a request_id")
    found = _index_natural_behavior_records(records, expected_requests)
    return [
        _add_natural_selection_positions(
            _natural_stimulus_from_record(
                request_id, found[request_id], selection_manifest, natural_state
            ),
            tokenizer,
            convention=convention,
            enable_thinking=enable_thinking,
        )
        for request_id in expected_requests
    ]


def write_natural_prefix_stimuli(  # noqa: PLR0913 - durable builder exposes its input identities
    records_path: Path,
    output_path: Path,
    selection_manifest_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    *,
    natural_state: str,
    convention: str,
    enable_thinking: bool,
) -> dict[str, Any]:
    """Build and write the private natural-prefix JSONL from retained behavior records."""
    manifest = load_natural_selection_manifest(selection_manifest_path)
    records_path = _private_selection_path(records_path)
    output_path = _private_selection_path(output_path)
    records = [
        cast("Mapping[str, Any]", json.loads(line))
        for line in records_path.read_text().splitlines()
        if line.strip()
    ]
    stimuli = build_natural_prefix_stimuli(
        records,
        tokenizer,
        manifest,
        natural_state=natural_state,
        convention=convention,
        enable_thinking=enable_thinking,
    )
    rendered = render_stimuli(
        tokenizer,
        stimuli,
        convention=convention,
        enable_thinking=enable_thinking,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": stimulus.stimulus_id,
                    "set": stimulus.stimulus_set,
                    "side": stimulus.side,
                    "pair_id": stimulus.pair_id,
                    "text": stimulus.text,
                    "assistant_prefix": stimulus.assistant_prefix,
                    "metadata": stimulus.metadata,
                },
                sort_keys=True,
            )
            + "\n"
            for stimulus in stimuli
        )
    )
    return {
        "records_file": str(records_path),
        "stimuli_file": str(output_path),
        "selection_manifest": manifest,
        "natural_state": natural_state,
        "stimuli_sha256": stimuli_digest(stimuli),
        "rendered_sha256": digest_of_strings(
            rendered[stimulus.stimulus_id] for stimulus in stimuli
        ),
        "tokenizer_identity": tokenizer_identity(tokenizer),
        "stimulus_render": convention,
        "enable_thinking": enable_thinking,
        "stimulus_ids": [stimulus.stimulus_id for stimulus in stimuli],
    }


def parse_natural_prefix_sources(raw_specs: Sequence[str]) -> dict[str, Path]:
    """Parse natural-prefix source files, optionally bound to a capture state or arm.

    A single unqualified path is retained as a convenience for one-cell captures.  A capture
    containing more than one state must use ``STATE_OR_ARM=PATH`` for every state so base and final
    rollouts cannot accidentally be captured from the same natural text.
    """
    if not raw_specs:
        return {}
    sources: dict[str, Path] = {}
    for raw in raw_specs:
        key, separator, raw_path = raw.partition("=")
        if separator and key and raw_path:
            if key in sources:
                raise ValueError(f"natural-prefix source repeats state or arm {key!r}")
            sources[key] = Path(raw_path)
            continue
        if len(raw_specs) != 1:
            raise ValueError(
                f"multiple natural-prefix sources must use STATE_OR_ARM=PATH; got {raw!r}"
            )
        sources["*"] = Path(raw)
    return sources


def natural_prefix_source_for_cell(
    cell: LadderCell, source_specs: Mapping[str, Path], *, n_non_base_cells: int
) -> Path:
    """Resolve the explicitly bound natural source for one capture cell."""
    if "*" in source_specs:
        if len(source_specs) != 1:
            raise ValueError(
                "an unqualified natural-prefix source cannot be combined with bindings"
            )
        return source_specs["*"]
    if cell.arm in source_specs:
        return source_specs[cell.arm]
    if cell.arm == BASE_ARM and "base" in source_specs:
        return source_specs["base"]
    if cell.arm != BASE_ARM and n_non_base_cells == 1 and "final" in source_specs:
        return source_specs["final"]
    raise ValueError(
        f"no natural-prefix source is bound to {cell.label}; provide {cell.arm}=PATH "
        "(or base=PATH/final=PATH for the two-cell read)"
    )


def _validate_natural_stimuli_state(
    stimuli: Sequence[Stimulus], selection_manifest: Mapping[str, Any], natural_state: str
) -> None:
    """Refuse a natural source whose retained rollouts belong to another ladder state."""
    if natural_state not in NATURAL_STATES:
        raise ValueError(
            f"natural_state must be one of {list(NATURAL_STATES)}, got {natural_state!r}"
        )
    expected_rollout_id = selection_manifest["rollout_ids_by_state"][natural_state]
    for stimulus in stimuli:
        actual_state = stimulus.metadata.get("natural_state")
        actual_rollout_id = stimulus.metadata.get("rollout_id")
        if actual_state != natural_state or actual_rollout_id != expected_rollout_id:
            raise ValueError(
                f"natural source stimulus {stimulus.stimulus_id!r} is bound to "
                f"state={actual_state!r}, rollout_id={actual_rollout_id!r}; expected "
                f"state={natural_state!r}, rollout_id={expected_rollout_id!r}"
            )


def parse_arm_spec(spec: str) -> tuple[str, Path]:
    """Split an `--arm name=run_dir` specification into its two halves."""
    name, separator, root = spec.partition("=")
    if not separator or not name or not root:
        raise ValueError(
            f"--arm expects 'name=run_dir', got {spec!r}; run_dir holds the checkpoint-* dirs."
        )
    return name, Path(root)


def parse_steps(raw: str | None) -> list[int] | None:
    """Parse a comma-separated step list, or None for 'every checkpoint present'."""
    if raw is None:
        return None
    return [int(part) for part in raw.split(",") if part.strip()]


def resolve_arm_cells(arm: str, run_dir: Path, steps: Sequence[int] | None) -> list[LadderCell]:
    """Return one arm's cells in training-step order, refusing a step that is not there.

    A missing checkpoint is fatal rather than skipped: a ladder quietly short one step reads later as
    a flat stretch of trend, which is indistinguishable from the finding.
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"arm {arm!r} run dir {run_dir} is not a directory.")
    found = {
        checkpoint_step(path): path
        for path in run_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    }
    wanted = sorted(found) if steps is None else list(steps)
    absent = [step for step in wanted if step not in found]
    if absent:
        raise FileNotFoundError(
            f"arm {arm!r} run dir {run_dir} has no checkpoints {absent}; it holds {sorted(found)}."
        )
    return [LadderCell(arm=arm, step=step, adapter_dir=found[step]) for step in wanted]


def template_prefix(tokenizer: PreTrainedTokenizerBase, *, enable_thinking: bool) -> str:
    """Return everything the chat template emits before a user turn's own content.

    Derived by rendering a sentinel rather than hardcoding a token, so the check built on it holds
    for any template in this family and cannot silently stop applying when one changes.
    """
    extras: dict[str, Any] = dict(resolve_chat_template_kwargs(tokenizer))
    rendered = cast(
        "str",
        tokenizer.apply_chat_template(
            [{"role": "user", "content": RENDER_PROBE_SENTINEL}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            **extras,
        ),
    )
    if RENDER_PROBE_SENTINEL not in rendered:
        raise ValueError(
            f"this tokenizer's chat template dropped the probe sentinel, so where a user turn's "
            f"content starts cannot be located and a pre-rendered corpus cannot be verified. "
            f"Rendered: {rendered!r}"
        )
    return rendered.split(RENDER_PROBE_SENTINEL)[0]


def assert_render_convention(
    tokenizer: PreTrainedTokenizerBase,
    stimuli: Sequence[Stimulus],
    *,
    convention: str,
    enable_thinking: bool,
) -> str:
    """Raise unless the corpus matches the render convention it is about to be captured under.

    Both mismatches are silent and both change the measured space. A `templated_here` corpus whose
    rows already carry the template gets a second user turn wrapped around the first, which puts the
    teacher-forced reasoning inside a user turn and leaves the assistant turn ending at an empty
    `<think>` -- the pooled position is then identical for both sides of every pair. A `verbatim`
    corpus whose rows are bare stems is read as a document rather than as a turn being answered.
    """
    if convention not in STIMULUS_RENDERS:
        raise ValueError(
            f"unknown --stimulus-render {convention!r}; expected {sorted(STIMULUS_RENDERS)}."
        )
    prefix = template_prefix(tokenizer, enable_thinking=enable_thinking)
    already = [stimulus.stimulus_id for stimulus in stimuli if prefix in stimulus.text]
    if convention == STIMULUS_RENDER_TEMPLATED:
        if any(stimulus.assistant_prefix for stimulus in stimuli) and not derive_prefilled_think(
            tokenizer, enable_thinking=enable_thinking
        ):
            raise ValueError(
                "this template does not open <think> inside the prompt, so an appended "
                "assistant_prefix would not sit inside the model's reasoning block -- it would "
                "become an assistant turn that states its reasoning as an answer. These stimuli were "
                "authored for the prefilled-think convention the games battery ran under; drop "
                "--no-thinking, or use a corpus with no assistant_prefix."
            )
        if already:
            raise ValueError(
                f"{len(already)} stimuli already carry this template's user-turn prefix "
                f"({already[:ERROR_EXAMPLE_COUNT]}), so templating them here would nest one user "
                f"turn inside another and pool a position identical for both sides of every pair. "
                f"Either emit untemplated stems, or capture with --stimulus-render verbatim. Prefix: "
                f"{prefix!r}"
            )
    else:
        unrendered = [
            stimulus.stimulus_id for stimulus in stimuli if not stimulus.text.startswith(prefix)
        ]
        if unrendered:
            raise ValueError(
                f"{len(unrendered)} stimuli do not begin with this template's user-turn prefix "
                f"({unrendered[:ERROR_EXAMPLE_COUNT]}), so a verbatim capture would feed the model a "
                f"bare document rather than a turn it is answering. Capture with --stimulus-render "
                f"{STIMULUS_RENDER_TEMPLATED} instead. Prefix: {prefix!r}"
            )
        carried = [stimulus.stimulus_id for stimulus in stimuli if stimulus.assistant_prefix]
        if carried:
            raise ValueError(
                f"{len(carried)} stimuli carry an assistant_prefix ({carried[:ERROR_EXAMPLE_COUNT]}) "
                f"under --stimulus-render verbatim, which appends nothing. The prefix has to be "
                f"inside 'text' already, or the reasoning it teacher-forces is silently dropped."
            )
    logger.info(f"render convention checked, {convention=} n={len(stimuli)} prefix={prefix!r}")
    return prefix


def render_stimuli(
    tokenizer: PreTrainedTokenizerBase,
    stimuli: Sequence[Stimulus],
    *,
    convention: str,
    enable_thinking: bool,
    include_assistant_prefix: bool = True,
) -> dict[str, str]:
    """Return the exact string each stimulus will be fed as, keyed by stimulus id.

    Under `templated_here` this is `template(stem) + assistant_prefix`: one user turn,
    `add_generation_prompt=True`, `enable_thinking` passed explicitly, plus whatever
    `resolve_chat_template_kwargs` pins -- the same call `games/dataset.py` makes at training time --
    and the prefix appended after the template's opening `<think>`, so the teacher-forced reasoning
    sits inside the assistant turn. Under `verbatim` the corpus already holds that string. Setting
    ``include_assistant_prefix=False`` renders the supplied prompt-end state, which is a separate
    measurement boundary and must never be confused with the teacher-forced state.
    """
    assert_render_convention(
        tokenizer, stimuli, convention=convention, enable_thinking=enable_thinking
    )
    if convention == STIMULUS_RENDER_VERBATIM:
        return {stimulus.stimulus_id: stimulus.text for stimulus in stimuli}
    extras: dict[str, Any] = dict(resolve_chat_template_kwargs(tokenizer))
    rendered: dict[str, str] = {}
    for stimulus in stimuli:
        templated = cast(
            "str",
            tokenizer.apply_chat_template(
                [{"role": "user", "content": stimulus.text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **extras,
            ),
        )
        rendered[stimulus.stimulus_id] = templated + (
            stimulus.assistant_prefix or "" if include_assistant_prefix else ""
        )
    if convention == STIMULUS_RENDER_VERBATIM and not include_assistant_prefix:
        raise ValueError(
            "prompt-end rendering requires untemplated stimuli with an assistant_prefix; a "
            "verbatim corpus already contains its teacher-forced suffix and cannot be split "
            "without a recorded token boundary."
        )
    return rendered


def assert_no_added_special_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> None:
    """Raise if this tokenizer prepends special tokens to an already-templated string.

    The capture tokenizes with the library default while the corpus already carries the template's
    own markers, so a family that adds a BOS on top would shift every position by one against the
    counts recorded here -- and against the pre-rendered convention the corpus was authored for.
    Verified false for Qwen3.5-2B on 2026-08-20; checked rather than remembered because it is a
    property of whichever tokenizer a run is actually handed.
    """
    with_special = cast("list[int]", tokenizer(text)["input_ids"])
    without_special = cast("list[int]", tokenizer(text, add_special_tokens=False)["input_ids"])
    if len(with_special) != len(without_special):
        raise ValueError(
            f"this tokenizer adds {len(with_special) - len(without_special)} special tokens to an "
            f"already-templated string, so the capture's tokenization does not match the corpus the "
            f"stimuli were authored as. Every recorded position would be off by that much."
        )


def token_counts(
    tokenizer: PreTrainedTokenizerBase, texts: Sequence[str], *, max_prompt_tokens: int
) -> list[int]:
    """Count each stimulus's tokens, refusing a corpus the capture would have to truncate.

    The budget is checked against the corpus's own floor before anything is tokenized, because a
    sub-floor window is an operator mistake and reads as one here, while the same mistake reported
    after counting reads as a corpus that is too long.
    """
    if max_prompt_tokens < MIN_CAPTURE_SEQ_LEN:
        raise ValueError(
            f"--max-prompt-tokens {max_prompt_tokens} is below the {MIN_CAPTURE_SEQ_LEN}-token floor "
            f"these stimuli are authored against. The two sides of a pair stay byte-identical until "
            f"token 326-415 of a 368-451 token string, so a window under the floor holds only their "
            f"shared prefix: both sides become the same string and every direction fitted on them is "
            f"exactly zero, which is a well-formed direction and raises nothing. Raise the budget "
            f"rather than the floor, which sits above the longest stimulus by construction."
        )
    if not texts:
        raise ValueError("no texts to count.")
    assert_no_added_special_tokens(tokenizer, texts[0])
    counts = [len(cast("list[int]", ids)) for ids in tokenizer(list(texts))["input_ids"]]
    over_budget = {index: count for index, count in enumerate(counts) if count > max_prompt_tokens}
    if over_budget:
        raise ValueError(
            f"{len(over_budget)} stimuli exceed the {max_prompt_tokens}-token budget "
            f"(worst {max(over_budget.values())}). Truncating would change what the model reads, so "
            f"the budget is a refusal rather than a clamp; raise --max-prompt-tokens deliberately."
        )
    return counts


def _token_ids(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """Tokenize one already-rendered string using the same no-extra-token path as capture."""
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if isinstance(ids, torch.Tensor):
        return [int(value) for value in ids.reshape(-1).tolist()]
    return [int(value) for value in cast("list[int]", ids)]


def contrast_boundaries(
    tokenizer: PreTrainedTokenizerBase,
    stimuli: Sequence[Stimulus],
    rendered: Mapping[str, str],
) -> tuple[ContrastBoundary, ...]:
    """Find each pair's first differing token in the exact rendered strings.

    The boundaries are measured before a capture window is applied. A matched pair whose boundary
    is outside that window can otherwise produce a perfectly valid zero direction.
    """
    by_pair: dict[str, dict[str, list[int]]] = {}
    for stimulus in stimuli:
        try:
            text = rendered[stimulus.stimulus_id]
        except KeyError as error:
            raise ValueError(
                f"rendered stimuli omit {stimulus.stimulus_id!r}; cannot check its contrast"
            ) from error
        by_pair.setdefault(stimulus.pair_id, {})[stimulus.side] = _token_ids(tokenizer, text)
    boundaries: list[ContrastBoundary] = []
    for pair_id, by_side in sorted(by_pair.items()):
        if set(by_side) != {"A", "B"}:
            raise ValueError(
                f"pair {pair_id!r} has sides {sorted(by_side)}, expected exactly A and B for "
                "a contrast check"
            )
        left, right = by_side["A"], by_side["B"]
        first = next(
            (
                index
                for index, (left_id, right_id) in enumerate(zip(left, right, strict=False))
                if left_id != right_id
            ),
            min(len(left), len(right)),
        )
        if first == len(left) == len(right):
            raise ValueError(f"pair {pair_id!r} has no rendered token contrast")
        boundaries.append(
            ContrastBoundary(
                pair_id=pair_id,
                first_differing_position=first,
                left_length=len(left),
                right_length=len(right),
            )
        )
    if not boundaries:
        raise ValueError("no matched pairs available for a contrast check")
    return tuple(boundaries)


def assert_contrast_survives_window(
    tokenizer: PreTrainedTokenizerBase,
    stimuli: Sequence[Stimulus],
    rendered: Mapping[str, str],
    *,
    window_start: int = 0,
    window_end: int | None = None,
) -> tuple[ContrastBoundary, ...]:
    """Refuse a crop/window that removes every token distinguishing either side of a pair."""
    if window_start < 0:
        raise ValueError(f"window_start must be non-negative, got {window_start}")
    if window_end is not None and window_end <= window_start:
        raise ValueError(
            f"window_end must be greater than window_start, got {window_start}:{window_end}"
        )
    boundaries = contrast_boundaries(tokenizer, stimuli, rendered)
    for boundary in boundaries:
        texts = {
            stimulus.side: _token_ids(tokenizer, rendered[stimulus.stimulus_id])
            for stimulus in stimuli
            if stimulus.pair_id == boundary.pair_id
        }
        end = window_end
        left = texts["A"][window_start:end]
        right = texts["B"][window_start:end]
        if left == right:
            raise ValueError(
                f"capture crop/window {window_start}:{end} removes the contrast for pair "
                f"{boundary.pair_id!r}; its first differing token is {boundary.first_differing_position}. "
                "Increase the window or retain a contrast-bearing suffix."
            )
    return boundaries


def _selection_from_stimulus(stimulus: Stimulus, token_count: int) -> NaturalPrefixSelection:
    """Validate a natural-prefix selection stored in private stimulus metadata."""
    raw_positions = stimulus.metadata.get("selected_positions")
    if raw_positions is None:
        raise ValueError(
            f"natural stimulus {stimulus.stimulus_id!r} has no selected_positions metadata; "
            "long natural traces must be reduced by an explicit selection rule"
        )
    if not isinstance(raw_positions, list) or not all(
        isinstance(position, int) and not isinstance(position, bool) for position in raw_positions
    ):
        raise ValueError(
            f"natural stimulus {stimulus.stimulus_id!r} selected_positions must be a list of "
            "integer token positions"
        )
    positions = tuple(raw_positions)
    if not positions:
        raise ValueError(f"natural stimulus {stimulus.stimulus_id!r} selected no positions")
    outside = [position for position in positions if not 0 <= position < token_count]
    if outside:
        raise ValueError(
            f"natural stimulus {stimulus.stimulus_id!r} selects positions {outside} outside its "
            f"{token_count}-token prompt"
        )
    if len(set(positions)) != len(positions):
        raise ValueError(
            f"natural stimulus {stimulus.stimulus_id!r} repeats selected token positions {positions}"
        )
    rule = stimulus.metadata.get("selection_rule")
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError(
            f"natural stimulus {stimulus.stimulus_id!r} has no selection_rule metadata; record "
            "how its positions were chosen"
        )
    return NaturalPrefixSelection(
        stimulus_id=stimulus.stimulus_id,
        positions=positions,
        selection_rule=rule,
        request_id=(
            None
            if stimulus.metadata.get("request_id") is None
            else str(stimulus.metadata["request_id"])
        ),
        rollout_id=(
            None
            if stimulus.metadata.get("rollout_id") is None
            else str(stimulus.metadata["rollout_id"])
        ),
        scenario_group=(
            None
            if stimulus.metadata.get("scenario_group") is None
            else str(stimulus.metadata["scenario_group"])
        ),
        natural_state=(
            None
            if stimulus.metadata.get("natural_state") is None
            else str(stimulus.metadata["natural_state"])
        ),
    )


@torch.no_grad()
def capture_selected_natural_activations(  # noqa: C901, PLR0912, PLR0913 - bounded capture loop validates and selects each batch
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    stimuli: Sequence[Stimulus],
    rendered: Mapping[str, str],
    *,
    layers: Sequence[int],
    batch_size: int = 1,
    selection_manifest: Mapping[str, Any] | None = None,
    max_prompt_tokens: int | None = None,
) -> tuple[dict[str, torch.Tensor], tuple[NaturalPrefixSelection, ...]]:
    """Capture only explicitly selected natural-prefix positions, discarding dense outputs promptly.

    ``capture_positionwise_activations`` is used for the one trunk forward, but its dense temporary
    tensors never leave this function. The returned mapping is stimulus id to
    ``[n_selected_positions, n_layers, hidden]`` CPU float32 tensors, and the selection metadata is
    returned beside it so a later read can tell which natural positions were retained.
    An explicit layer subset is required because retaining every layer of a long rollout is the
    dense-trace failure this helper exists to avoid.
    """
    if not layers:
        raise ValueError("natural-prefix capture requires an explicit non-empty layer subset")
    if batch_size < 1:
        raise ValueError(f"natural-prefix capture batch_size must be positive, got {batch_size}")
    if max_prompt_tokens is not None and max_prompt_tokens < 1:
        raise ValueError(
            f"natural-prefix capture max_prompt_tokens must be positive, got {max_prompt_tokens}"
        )
    if selection_manifest is not None:
        declared_layers = tuple(int(layer) for layer in selection_manifest["layers"])
        if tuple(layers) != declared_layers:
            raise ValueError(
                f"natural-prefix layers {tuple(layers)} do not match the predeclared manifest "
                f"layers {declared_layers}"
            )
    if tokenizer.padding_side != "right":
        raise ValueError(
            f"natural-prefix capture needs right padding, got {tokenizer.padding_side!r}; "
            "position ids would otherwise shift between padded rows"
        )
    selected: dict[str, torch.Tensor] = {}
    selections: list[NaturalPrefixSelection] = []
    for start in range(0, len(stimuli), batch_size):
        stimulus_batch = stimuli[start : start + batch_size]
        texts = []
        batch_selections: list[NaturalPrefixSelection] = []
        for stimulus in stimulus_batch:
            text = rendered.get(stimulus.stimulus_id)
            if text is None:
                raise ValueError(f"rendered stimuli omit {stimulus.stimulus_id!r}")
            token_count = len(_token_ids(tokenizer, text))
            if max_prompt_tokens is not None and token_count > max_prompt_tokens:
                raise ValueError(
                    f"natural stimulus {stimulus.stimulus_id!r} has {token_count} tokens, over "
                    f"the exact {max_prompt_tokens}-token capture budget; increase the budget "
                    "explicitly rather than cropping the selected prefix"
                )
            texts.append(text)
            selection = _selection_from_stimulus(stimulus, token_count)
            if selection_manifest is not None:
                _validate_natural_capture_selection(stimulus, selection, selection_manifest)
            batch_selections.append(selection)
        encoded = tokenizer(texts, return_tensors="pt", padding=True)
        positionwise = capture_positionwise_activations(
            model,  # pyright: ignore[reportArgumentType] -- capture hooks accept the loaded base model
            encoded["input_ids"],
            encoded["attention_mask"],
            layers=layers,
        )
        try:
            for row, selection in enumerate(batch_selections):
                positions = torch.tensor(selection.positions, dtype=torch.long)
                selected[selection.stimulus_id] = torch.stack(
                    [positionwise[layer][row, positions, :] for layer in layers], dim=1
                ).cpu()
            selections.extend(batch_selections)
        finally:
            del positionwise
    return selected, tuple(selections)


def _validate_natural_capture_selection(
    stimulus: Stimulus,
    selection: NaturalPrefixSelection,
    selection_manifest: Mapping[str, Any],
) -> None:
    """Check one selected stimulus against the frozen natural capture contract."""
    if selection.selection_rule != selection_manifest["selection_rule"]:
        raise ValueError(
            f"natural stimulus {stimulus.stimulus_id!r} selection rule differs from "
            "the predeclared natural selection manifest"
        )
    natural_state = stimulus.metadata["natural_state"]
    expected_rollout_id = selection_manifest["rollout_ids_by_state"][natural_state]
    if stimulus.metadata.get("rollout_id") != expected_rollout_id:
        raise ValueError(
            f"natural stimulus {stimulus.stimulus_id!r} has rollout_id "
            f"{stimulus.metadata.get('rollout_id')!r}; state {natural_state!r} requires "
            f"{expected_rollout_id!r}"
        )
    if stimulus.metadata.get("scenario_group") not in selection_manifest["scenario_groups"]:
        raise ValueError(
            f"natural stimulus {stimulus.stimulus_id!r} has unregistered scenario_group "
            f"{stimulus.metadata.get('scenario_group')!r}"
        )


def capture_supplied_prefix_states(  # noqa: PLR0913 - explicit model, tokenizer, rendering and pooling controls
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    stimuli: Sequence[Stimulus],
    *,
    convention: str,
    enable_thinking: bool,
    poolings: Sequence[str] = ("last",),
    batch_size: int = 1,
) -> dict[str, dict[str, torch.Tensor]]:
    """Capture prompt-end and teacher-forced states for the same supplied-prefix stimuli.

    The prompt-end pass stops after the rendered user turn. The teacher-forced pass appends each
    row's short ``assistant_prefix``. Keeping the two boundaries under explicit names prevents a
    continuation state from being mistaken for the prompt-end representation in a paired read.
    """
    if convention != STIMULUS_RENDER_TEMPLATED:
        raise ValueError(
            "supplied-prefix states require templated_here stimuli so the prompt-end boundary can "
            "be reconstructed; verbatim rows must carry an explicit boundary in their metadata"
        )
    if any(stimulus.assistant_prefix is None for stimulus in stimuli):
        missing = [
            stimulus.stimulus_id for stimulus in stimuli if stimulus.assistant_prefix is None
        ]
        raise ValueError(
            f"supplied-prefix capture requires assistant_prefix on every row; missing {missing[:5]}"
        )
    prompt_end = render_stimuli(
        tokenizer,
        stimuli,
        convention=convention,
        enable_thinking=enable_thinking,
        include_assistant_prefix=False,
    )
    teacher_forced = render_stimuli(
        tokenizer,
        stimuli,
        convention=convention,
        enable_thinking=enable_thinking,
        include_assistant_prefix=True,
    )
    return {
        "prompt_end": capture_pooled_matrix(
            model,
            tokenizer,
            [prompt_end[stimulus.stimulus_id] for stimulus in stimuli],
            poolings=poolings,
            batch_size=batch_size,
        ),
        "teacher_forced": capture_pooled_matrix(
            model,
            tokenizer,
            [teacher_forced[stimulus.stimulus_id] for stimulus in stimuli],
            poolings=poolings,
            batch_size=batch_size,
        ),
    }


def capture_pooled_matrix(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    poolings: Sequence[str],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Capture one stimulus set every requested way: pooling -> `[n_texts, n_layers, hidden]` float32.

    A thin stack over `capture_pooled_activations_multi`, which returns one `[n, hidden]` tensor per
    decoder-block output per pooling. Stacking on the layer axis here is what makes a cell one tensor
    per (set, pooling) rather than one per layer, and it pins the layer axis to sorted block order --
    the `post_block` convention the cell identity records.

    Every pooler reads the same hidden state inside one forward hook, so asking for two poolings costs
    one pass over the corpus and returns bit-for-bit what two single-pooling passes returned (the
    forward is deterministic and each pooler's arithmetic on the same float32 upcast is unchanged;
    `reward_hacking.interp.directions` pins that equality against a stash-then-pool reference). It
    used to cost one forward per pooling -- 10-20 s per cell at 2B, ~15 s per pooling at 9B -- on the
    argument that a both-at-once hook would be a second capture path; it is now the only capture path,
    with the one-pooling call spelled as the multi call over one name.
    """
    by_pooling = capture_pooled_activations_multi(
        model,  # pyright: ignore[reportArgumentType]  # annotated as the factory class, wants the instance
        tokenizer,  # pyright: ignore[reportArgumentType]  # same: AutoTokenizer names the factory
        list(texts),
        poolings=list(poolings),
        batch_size=batch_size,
    )
    return {
        pooling: torch.stack([by_layer[layer] for layer in sorted(by_layer)], dim=1)
        for pooling, by_layer in by_pooling.items()
    }


def capture_cell(  # noqa: PLR0913 - a cell is a model, a tokenizer, a corpus and the capture knobs
    cell: LadderCell,
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    grouped: dict[str, list[Stimulus]],
    rendered: dict[str, str],
    counts_by_id: dict[str, int],
    poolings: Sequence[str],
    batch_size: int,
) -> tuple[dict[str, Any], dict[tuple[str, str], torch.Tensor]]:
    """Run every stimulus set at one model state and return its row indices and activations."""
    rows: dict[str, Any] = {}
    activations: dict[tuple[str, str], torch.Tensor] = {}
    for stimulus_set, members in sorted(grouped.items()):
        texts = [rendered[member.stimulus_id] for member in members]
        rows[stimulus_set] = row_index_for(
            members, [counts_by_id[member.stimulus_id] for member in members]
        )
        by_pooling = capture_pooled_matrix(
            model, tokenizer, texts, poolings=poolings, batch_size=batch_size
        )
        for pooling in poolings:
            matrix = by_pooling[pooling]
            activations[stimulus_set, pooling] = matrix
            logger.info(
                f"captured, cell={cell.label} {stimulus_set=} {pooling=} shape={list(matrix.shape)}"
            )
    return rows, activations


def build_identity(  # noqa: PLR0913 - an identity is exactly these fields, by construction
    model: PreTrainedModel,
    *,
    base_model: str,
    digest: str,
    rendered_digest: str,
    stimulus_render: str,
    batch_size: int,
    compute_dtype: str,
    store_dtype: str,
    tokenizer_identity: str = "unspecified",
    kernel_identity: str = "unspecified",
    capture_prefix_states: bool = False,
    prompt_end_rendered_sha256: str = "unspecified",
    teacher_forced_rendered_sha256: str = "unspecified",
    natural_prefix_layers: Sequence[int] = (),
    natural_selection_manifest_sha256: str = "",
    natural_prefix_stimuli_sha256: str = "",
    natural_prefix_rendered_sha256: str = "",
) -> CellIdentity:
    """Read the measured space off the loaded model rather than off a config the caller passed."""
    text_config = getattr(model.config, "text_config", model.config)
    return CellIdentity(
        base_model=base_model,
        stimuli_sha256=digest,
        rendered_sha256=rendered_digest,
        layer_convention=LAYER_CONVENTION_POST_BLOCK,
        n_layers=int(text_config.num_hidden_layers),
        hidden_size=int(text_config.hidden_size),
        batch_size=batch_size,
        compute_dtype=compute_dtype,
        store_dtype=store_dtype,
        stimulus_render=stimulus_render,
        tokenizer_identity=tokenizer_identity,
        kernel_identity=kernel_identity,
        capture_prefix_states=capture_prefix_states,
        prompt_end_rendered_sha256=prompt_end_rendered_sha256,
        teacher_forced_rendered_sha256=teacher_forced_rendered_sha256,
        natural_prefix_layers=tuple(natural_prefix_layers),
        natural_selection_manifest_sha256=natural_selection_manifest_sha256,
        natural_prefix_stimuli_sha256=natural_prefix_stimuli_sha256,
        natural_prefix_rendered_sha256=natural_prefix_rendered_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    """CLI for the checkpoint-ladder capture driver."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stimuli", type=Path, required=False, default=None, help="JSONL of stimulus rows."
    )
    parser.add_argument(
        "--cooperation-constructs",
        action="store_true",
        help="Load and validate the private two-construct corpus, including its authored grouped split.",
    )
    parser.add_argument(
        "--reserved-identities",
        type=Path,
        default=None,
        help="JSON {group_ids, pair_ids} reserved by training/evaluation corpora.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=False,
        default=None,
        help="Capture root to write cells under.",
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=RUN_DIR",
        help="Arm name and the run dir holding its checkpoint-* directories. Repeatable.",
    )
    parser.add_argument(
        "--steps", default=None, help="Comma-separated checkpoint steps; default every one present."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Do not capture the un-adapted base, which is the anchor every drift read needs.",
    )
    parser.add_argument(
        "--poolings",
        default=",".join(DEFAULT_POOLINGS),
        help="Comma-separated poolings to capture per stimulus set.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=DEFAULT_MAX_PROMPT_TOKENS,
        help=f"Per-stimulus token budget, a refusal rather than a clamp, floored at "
        f"{MIN_CAPTURE_SEQ_LEN}: under that a matched pair's two sides collapse to their shared stem.",
    )
    parser.add_argument("--store-dtype", choices=sorted(STORE_DTYPES), default="float32")
    parser.add_argument("--compute-dtype", choices=sorted(COMPUTE_DTYPES), default="bfloat16")
    parser.add_argument(
        "--limit-stimuli",
        type=int,
        default=None,
        help="Capture only the first N stimuli. Changes the corpus digest, so a smoke cell cannot "
        "be compared against a full one.",
    )
    parser.add_argument(
        "--stimulus-render",
        choices=sorted(STIMULUS_RENDERS),
        default=STIMULUS_RENDER_TEMPLATED,
        help="Whether this driver chat-templates each stem and appends its assistant_prefix "
        "(templated_here, what games.interp_stimuli emits) or feeds 'text' as-is (verbatim).",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Render and verify against the thinking-off template; the arms trained with it on.",
    )
    parser.add_argument(
        "--capture-prefix-states",
        action="store_true",
        help="Also persist supplied prompt-end and teacher-forced prefix pooled states.",
    )
    parser.add_argument(
        "--natural-prefix-layers",
        default=None,
        help="Comma-separated decoder layers for explicit natural-prefix position captures; "
        "selected positions must be present in each stimulus metadata.",
    )
    parser.add_argument(
        "--natural-prefix-stimuli",
        action="append",
        default=[],
        metavar="STATE_OR_ARM=PATH",
        help="Separate natural-rollout prefix JSONL. Bind one source per state/arm with "
        "STATE_OR_ARM=PATH; a single unqualified path is allowed for a one-cell capture.",
    )
    parser.add_argument(
        "--natural-selection-manifest",
        type=Path,
        default=None,
        help="Predeclared state-specific rollout IDs, scenario groups, rule and layer subset.",
    )
    parser.add_argument(
        "--natural-state",
        choices=NATURAL_STATES,
        default=None,
        help="Ladder state represented by --records when building natural-prefix stimuli.",
    )
    parser.add_argument(
        "--build-natural-prefix-stimuli",
        action="store_true",
        help="Build private natural-prefix JSONL from retained behavior records and exit before model loading.",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="Retained behavior JSONL for --build-natural-prefix-stimuli.",
    )
    parser.add_argument(
        "--natural-prefix-out",
        type=Path,
        default=None,
        help="Output JSONL for --build-natural-prefix-stimuli.",
    )
    parser.add_argument(
        "--deadline", default=None, help="ISO-8601 UTC instant after which no new cell starts."
    )
    return parser


def resolve_cells(args: argparse.Namespace) -> list[LadderCell]:
    """Build the run's agenda: the base anchor, then each arm's checkpoints in step order."""
    steps = parse_steps(args.steps)
    cells: list[LadderCell] = []
    if not args.skip_base:
        cells.append(LadderCell(arm=BASE_ARM, step=BASE_STEP, adapter_dir=None))
    for spec in cast("list[str]", args.arm):
        name, run_dir = parse_arm_spec(spec)
        cells.extend(resolve_arm_cells(name, run_dir, steps))
    if not cells:
        raise ValueError("nothing to capture: pass at least one --arm, or drop --skip-base.")
    assert_one_adapter_config([cell.adapter_dir for cell in cells if cell.adapter_dir is not None])
    return cells


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915 - linear CLI driver owns model/capture lifecycle
    """Capture every cell on the agenda, writing each one as it finishes."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)
    if args.build_natural_prefix_stimuli:
        if (
            args.records is None
            or args.natural_selection_manifest is None
            or args.natural_prefix_out is None
            or args.natural_state is None
        ):
            raise ValueError(
                "--build-natural-prefix-stimuli requires --records, --natural-selection-manifest, "
                "--natural-prefix-out, and --natural-state"
            )
        tokenizer = cast(
            "PreTrainedTokenizerBase",
            AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True),
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        report = write_natural_prefix_stimuli(
            args.records,
            args.natural_prefix_out,
            args.natural_selection_manifest,
            tokenizer,
            natural_state=args.natural_state,
            convention=args.stimulus_render,
            enable_thinking=not args.no_thinking,
        )
        report["records_sha256"] = sha256_of_file(args.records)
        report["selection_manifest_path"] = str(args.natural_selection_manifest)
        sidecar = args.natural_prefix_out.with_suffix(
            args.natural_prefix_out.suffix + ".manifest.json"
        )
        sidecar.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        logger.info("natural-prefix stimuli built: %s", json.dumps(report, sort_keys=True))
        return 0
    if args.stimuli is None or args.out_dir is None:
        raise ValueError("capture mode requires --stimuli and --out-dir")
    poolings = [part.strip() for part in str(args.poolings).split(",") if part.strip()]
    unknown = sorted(set(poolings) - set(POOLERS))
    if unknown:
        raise ValueError(f"unknown poolings {unknown}; expected from {sorted(POOLERS)}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deadline = None if args.deadline is None else dt.datetime.fromisoformat(args.deadline)

    reserved_groups, external_pair_ids = load_capture_reservations(args.reserved_identities)
    if args.cooperation_constructs:
        stimuli = load_private_construct_stimuli(
            args.stimuli,
            reserved_groups=reserved_groups,
            external_pair_ids=external_pair_ids,
        )
        construct_splits = construct_split_payload(stimuli)
    else:
        stimuli = load_stimuli(args.stimuli)
        construct_splits = None
    if args.limit_stimuli is not None:
        stimuli = stimuli[: args.limit_stimuli]
        logger.warning(
            f"capturing only the first {args.limit_stimuli} stimuli; this is a different corpus and "
            f"gets a different digest, so these cells cannot be compared against full ones"
        )
    digest = stimuli_digest(stimuli)
    grouped = group_by_set(stimuli)
    cells = resolve_cells(args)

    natural_source_specs = parse_natural_prefix_sources(args.natural_prefix_stimuli)
    natural_stimuli_by_cell: dict[str, list[Stimulus]] = {}
    natural_source_paths_by_cell: dict[str, Path] = {}
    natural_selection_manifest: dict[str, Any] | None = None
    natural_rendered_by_cell: dict[str, dict[str, str]] = {}
    natural_digests_by_cell: dict[str, str] = {}
    natural_rendered_digests_by_cell: dict[str, str] = {}
    natural_layers: tuple[int, ...] = ()
    if natural_source_specs:
        if "*" in natural_source_specs and len(cells) != 1:
            raise ValueError(
                "a natural-prefix source without a state/arm binding is only valid for a "
                "one-cell capture; bind base and final sources separately"
            )
        if args.natural_selection_manifest is None:
            raise ValueError(
                "--natural-prefix-stimuli requires --natural-selection-manifest so position "
                "selection is predeclared before capture"
            )
        natural_selection_manifest = load_natural_selection_manifest(
            args.natural_selection_manifest
        )
        natural_layers = tuple(int(layer) for layer in natural_selection_manifest["layers"])
        if args.natural_prefix_layers is not None:
            requested_layers = tuple(
                int(part.strip())
                for part in str(args.natural_prefix_layers).split(",")
                if part.strip()
            )
            if requested_layers != natural_layers:
                raise ValueError(
                    f"--natural-prefix-layers {requested_layers} differs from the predeclared "
                    f"manifest layers {natural_layers}"
                )
        n_non_base_cells = sum(cell.arm != BASE_ARM for cell in cells)
        for cell in cells:
            source_path = natural_prefix_source_for_cell(
                cell,
                natural_source_specs,
                n_non_base_cells=n_non_base_cells,
            )
            source_path = _private_selection_path(source_path)
            natural_source_paths_by_cell[cell.label] = source_path
            expected_state = "base" if cell.arm == BASE_ARM else "final"
            natural_stimuli = load_stimuli(source_path)
            _validate_natural_stimuli_state(
                natural_stimuli, natural_selection_manifest, expected_state
            )
            natural_stimuli_by_cell[cell.label] = natural_stimuli
    elif args.natural_prefix_layers is not None or args.natural_selection_manifest is not None:
        raise ValueError(
            "--natural-prefix-layers and --natural-selection-manifest require "
            "--natural-prefix-stimuli"
        )

    tokenizer = cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True),
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = render_stimuli(
        tokenizer,
        stimuli,
        convention=args.stimulus_render,
        enable_thinking=not args.no_thinking,
    )
    rendered_digest = digest_of_strings(rendered[stimulus.stimulus_id] for stimulus in stimuli)
    for cell in cells:
        natural_stimuli = natural_stimuli_by_cell.get(cell.label)
        if natural_stimuli is None:
            continue
        natural_rendered = render_stimuli(
            tokenizer,
            natural_stimuli,
            convention=args.stimulus_render,
            enable_thinking=not args.no_thinking,
        )
        natural_rendered_by_cell[cell.label] = natural_rendered
        natural_digests_by_cell[cell.label] = stimuli_digest(natural_stimuli)
        natural_rendered_digests_by_cell[cell.label] = digest_of_strings(
            natural_rendered[stimulus.stimulus_id] for stimulus in natural_stimuli
        )
        token_counts(
            tokenizer,
            [natural_rendered[stimulus.stimulus_id] for stimulus in natural_stimuli],
            max_prompt_tokens=args.max_prompt_tokens,
        )
    counts = token_counts(
        tokenizer,
        [rendered[stimulus.stimulus_id] for stimulus in stimuli],
        max_prompt_tokens=args.max_prompt_tokens,
    )
    counts_by_id = {
        stimulus.stimulus_id: count for stimulus, count in zip(stimuli, counts, strict=True)
    }
    if args.cooperation_constructs:
        assert_contrast_survives_window(
            tokenizer,
            stimuli,
            rendered,
            window_start=0,
            window_end=args.max_prompt_tokens,
        )
    logger.info(
        f"corpus ready, digest={short_digest(digest)} rendered={short_digest(rendered_digest)} "
        f"n={len(stimuli)} sets={ {name: len(members) for name, members in sorted(grouped.items())} } "
        f"tokens_min={min(counts)} tokens_max={max(counts)} device={device} "
        f"example={rendered[stimuli[0].stimulus_id][:120]!r}"
    )

    # Before the load, which imports the modeling module and freezes each kernel's dispatch.
    kernel_bridge = bridge_and_check_decode_kernel()
    model = load_adapter_base(
        args.base_model, dtype=COMPUTE_DTYPES[args.compute_dtype], device=device
    )
    kernels_bound = bound_deltanet_kernels()
    deltanet_kernel = prefill_deltanet_kernels(kernels_bound)
    identity = build_identity(
        model,
        base_model=args.base_model,
        digest=digest,
        rendered_digest=rendered_digest,
        stimulus_render=args.stimulus_render,
        batch_size=args.batch_size,
        compute_dtype=args.compute_dtype,
        store_dtype=args.store_dtype,
        tokenizer_identity=tokenizer_identity(tokenizer),
        kernel_identity=kernel_identity(deltanet_kernel),
        capture_prefix_states=args.capture_prefix_states,
        prompt_end_rendered_sha256=(
            digest_of_strings(
                render_stimuli(
                    tokenizer,
                    stimuli,
                    convention=args.stimulus_render,
                    enable_thinking=not args.no_thinking,
                    include_assistant_prefix=False,
                )[stimulus.stimulus_id]
                for stimulus in stimuli
            )
            if args.capture_prefix_states
            else "unspecified"
        ),
        teacher_forced_rendered_sha256=(
            rendered_digest if args.capture_prefix_states else "unspecified"
        ),
        natural_prefix_layers=natural_layers,
        natural_selection_manifest_sha256=(
            "" if natural_selection_manifest is None else natural_selection_manifest["sha256"]
        ),
        # Natural rollout text is state-specific. Its source and rendered digests live in each
        # standalone natural-prefix manifest; keeping them out of the shared ladder identity lets
        # base and final use matched request IDs with different retained rollout text.
        natural_prefix_stimuli_sha256="",
        natural_prefix_rendered_sha256="",
    )
    ladder: dict[str, Any] = {
        "identity": identity.to_payload(),
        "git_sha": git_sha(),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "stimuli_file": str(args.stimuli),
        "requested_cells": [cell.label for cell in cells],
        "poolings": poolings,
        "cooperation_constructs": args.cooperation_constructs,
        "construct_splits": construct_splits,
        "reserved_groups": list(reserved_groups),
        "reserved_pair_ids": list(external_pair_ids),
        "natural_prefix_stimuli_file": {
            label: str(path) for label, path in sorted(natural_source_paths_by_cell.items())
        },
        "natural_prefix_stimuli_sha256": dict(sorted(natural_digests_by_cell.items())),
        "natural_prefix_rendered_sha256": dict(sorted(natural_rendered_digests_by_cell.items())),
        "natural_selection_manifest": natural_selection_manifest,
        # No ladder-level DELTANET_KERNEL_FIELD, for the reason the module docstring gives.
        "deltanet_kernel_bridge": kernel_bridge,
        "deltanet_kernels_bound": kernels_bound,
        "cells": {},
    }

    adapted: PeftModel | None = None
    for cell in cells:
        if deadline is not None and dt.datetime.now(tz=dt.UTC) >= deadline:
            logger.warning(f"deadline {args.deadline} reached; stopping before {cell.label}")
            ladder["stopped_reason"] = f"deadline reached before {cell.label}"
            break
        cell_dir = step_dir(args.out_dir, cell.arm, cell.step)
        if (cell_dir / CELL_MANIFEST_FILENAME).is_file():
            prefix_ready = (
                not args.capture_prefix_states
                or (cell_dir / "prefix-activations.safetensors").is_file()
            )
            natural_ready = (
                cell.label not in natural_stimuli_by_cell
                or (cell_dir / "natural-prefix" / "natural-prefix-manifest.json").is_file()
            )
            if prefix_ready and natural_ready:
                logger.info(f"skipping {cell.label}: an earlier attempt already finished it")
                ladder["cells"][cell.label] = {"skipped": "already on disk"}
                continue
            logger.warning(
                f"re-capturing {cell.label}: the existing cell is missing requested auxiliary "
                "capture artifacts"
            )
        applied: int | None = None
        adapter_sha: str | None = None
        if cell.adapter_dir is not None:
            attached = attach_adapter(model, cell.adapter_dir, args.base_model, existing=adapted)
            adapted, applied = attached.peft_model, attached.applied_adapter_weights
            adapter_sha = sha256_of_file(cell.adapter_dir / ADAPTER_WEIGHTS_FILENAME)
        started = time.time()
        rows, activations = capture_cell(
            cell,
            model=model,
            tokenizer=tokenizer,
            grouped=grouped,
            rendered=rendered,
            counts_by_id=counts_by_id,
            poolings=poolings,
            batch_size=args.batch_size,
        )
        prefix_activations = None
        if args.capture_prefix_states:
            prefix_activations = {
                (state, pooling): matrix
                for state, by_pooling in capture_supplied_prefix_states(
                    model,
                    tokenizer,
                    stimuli,
                    convention=args.stimulus_render,
                    enable_thinking=not args.no_thinking,
                    poolings=poolings,
                    batch_size=args.batch_size,
                ).items()
                for pooling, matrix in by_pooling.items()
            }
        natural_prefix_activations = None
        natural_selections = ()
        natural_stimuli = natural_stimuli_by_cell.get(cell.label)
        if natural_stimuli is not None:
            natural_prefix_activations, natural_selections = capture_selected_natural_activations(
                model,
                tokenizer,
                natural_stimuli,
                natural_rendered_by_cell[cell.label],
                layers=natural_layers,
                batch_size=args.batch_size,
                selection_manifest=natural_selection_manifest,
                max_prompt_tokens=args.max_prompt_tokens,
            )
        seconds = round(time.time() - started, 2)
        write_cell(
            cell_dir,
            arm=cell.arm,
            step=cell.step,
            identity=identity,
            rows=rows,
            activations=activations,
            applied_adapter_weights=applied,
            adapter_weights_sha256=adapter_sha,
            provenance={
                "git_sha": ladder["git_sha"],
                "torch_version": ladder["torch_version"],
                "device": ladder["device"],
                "stimuli_file": ladder["stimuli_file"],
                DELTANET_KERNEL_FIELD: deltanet_kernel,
                "adapter_dir": None if cell.adapter_dir is None else str(cell.adapter_dir),
                "adapter_config": None
                if cell.adapter_dir is None
                else adapter_config_identity(cell.adapter_dir),
                "seconds": seconds,
                "captured_at": dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            stimulus_metadata={
                stimulus.stimulus_id: dict(stimulus.metadata)
                for stimulus in stimuli
                if stimulus.metadata
            },
            prefix_activations=prefix_activations,
            prefix_stimulus_ids=(
                tuple(stimulus.stimulus_id for stimulus in stimuli) if prefix_activations else None
            ),
        )
        if natural_stimuli is not None:
            write_natural_prefix_capture(
                cell_dir / "natural-prefix",
                identity=identity,
                stimuli=natural_stimuli,
                activations=cast("dict[str, torch.Tensor]", natural_prefix_activations),
                selection_records=[
                    {
                        **asdict(selection),
                        **{
                            field: stimulus.metadata[field]
                            for field in (
                                "request_id",
                                "rollout_id",
                                "scenario_group",
                                "natural_state",
                            )
                            if field in stimulus.metadata
                        },
                    }
                    for stimulus, selection in zip(natural_stimuli, natural_selections, strict=True)
                ],
                selection_manifest=cast("dict[str, Any]", natural_selection_manifest),
                rendered_sha256=natural_rendered_digests_by_cell[cell.label],
                natural_state="base" if cell.arm == BASE_ARM else "final",
                arm=cell.arm,
                step=cell.step,
                applied_adapter_weights=applied,
                adapter_weights_sha256=adapter_sha,
            )
        ladder["cells"][cell.label] = {"seconds": seconds, "applied_adapter_weights": applied}
        write_ladder_manifest(args.out_dir, ladder)
        logger.info(f"cell done, {cell.label} {seconds=} {applied=}")

    write_ladder_manifest(args.out_dir, ladder)
    logger.info(f"capture finished, cells={len(ladder['cells'])} out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
