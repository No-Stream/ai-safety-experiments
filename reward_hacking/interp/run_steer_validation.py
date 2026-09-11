"""Stage driver for the steering instrument-validation run: fit, screen, steer, read out.

Seven stages, each its own process invocation with its own artifacts, so a spot reclaim or an
expired deadline costs the in-flight stage and nothing else. Read
:mod:`reward_hacking.interp.steer_validation` first -- it carries the design and the reasoning; this
module is the sequencing, the artifacts and the command line.

    isolation           rig integrity: does one condition leak recurrent state into the next
    fit                 generate the symmetric instructed pairs, fit a direction at every layer
    screen-layers       teacher-forced preference shift at every layer (cheap, stated-preference)
    screen-strength     signed strength sweep x three position arms x the full control set
    behaviour           the behavioural read: non-thinking greedy generation, deterministic scoring
    behaviour-thinking  the same at the chosen cells with thinking on and no token cap
    report              per-item tables over whatever is on disk (CPU only, no model)

**Stages hand off through artifacts, and the later ones select their own cells.** ``screen-strength``
reads ``screen_layers.jsonl`` and takes the layers whose real arm most exceeds the control band the
same screen measured at that layer, unless told otherwise; ``behaviour`` does the same from
``screen_strength.jsonl``. The excess rather than the raw shift, because the largest of thirty-two
layers that all did nothing is a draw from a flat distribution (see :func:`top_layers_from_screen`).
Nothing is anchored on a layer chosen in an earlier run, which is how the previous causal tier ended
up steering the single worst-decoding layer of thirty-two.

**Every stage writes its summary LAST.** A stage's ``*_summary.json`` is its completion marker: the
JSONL is appended as records are produced, so its presence says the stage started, never that it
finished. Each summary carries what the stage attempted, what it produced and what it skipped, so a
zero in a readout arrives with its denominator.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from reward_hacking.interp.directions import load_model_and_tokenizer
from reward_hacking.interp.generation_capture import (
    capture_record_activations,
    generate_response,
    pool_positions,
    resolved_sampler,
    response_positions,
)
from reward_hacking.interp.steer_validation import (
    ARM_REAL,
    CONTROL_ARMS,
    DEFAULT_RHOS,
    DEFAULT_SCORE_BATCH_ROWS,
    GENERATION_SEED_STRIDE,
    LANGUAGE_INSTRUCTIONS,
    MAX_LAYER_FRACTION,
    POSITION_ARMS,
    SCRIPT_HAN,
    SCRIPT_LATIN,
    TARGET_SCRIPTS,
    EncodedScoringBatch,
    GapReadout,
    GapRecord,
    PositionArm,
    ResidualScale,
    ScoredContinuation,
    SteeringCell,
    alpha_from_relative_displacement,
    arm_family,
    build_prompts,
    cache_isolation_report,
    control_directions,
    encode_scoring_batch,
    isolation_verdict,
    kl_divergence_rows,
    non_thinking_sampling,
    residual_scale_stats,
    run_behavioural_cell,
    score_teacher_forced_gap,
    script_score,
    thinking_sampling,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from _typeshed import DataclassInstance
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reward_hacking.model_backend import SamplingConfig

logger = logging.getLogger(__name__)

ISOLATION_ARTIFACT = "isolation_checks.json"
DIRECTIONS_ARTIFACT = "directions.pt"
SCALES_ARTIFACT = "residual_scales.json"
CONTINUATIONS_ARTIFACT = "continuations.json"
FIT_GENERATIONS_ARTIFACT = "fit_generations.jsonl"
SCREEN_LAYERS_ARTIFACT = "screen_layers.jsonl"
SCREEN_STRENGTH_ARTIFACT = "screen_strength.jsonl"
BEHAVIOUR_ARTIFACT = "behaviour.jsonl"
BEHAVIOUR_THINKING_ARTIFACT = "behaviour_thinking.jsonl"
REPORT_ARTIFACT = "report.json"

_QUARTILE_MIN_N = 4
"""Below four values ``statistics.quantiles`` cannot form quartiles, so the range stands in."""

CellKey = tuple[int, str, str, float]
"""(layer, position_arm, arm, rho): what identifies a behavioural cell across runs and resumes."""

PICK_MIN_SCORABLE_FRACTION = 0.75
"""``pick-cell`` eligibility floor: the share of a cell's rows that must carry a scorable ratio.

Below it a cell's median rests on a handful of rows -- the defect this exists for was a cell with
two scorable rows winning a grid on the median of those two. Deliberately a fraction of the cell's
planned depth, not an absolute count, so it scales with ``--n-behaviour-prompts``."""

_SPEARMAN_MIN_N = 3
"""A rank correlation over two points is always +/-1, which says nothing about monotonicity."""

_COMPLIANCE_RATIO = 0.5
"""Script-ratio boundary for "this generation followed its language instruction".

A midpoint, not a tuned threshold: a compliant Chinese answer scores near 1.0 and a compliant
English one near 0.0, so anything in between is a mixed response that the fit should count as
non-compliant whichever side it came from.
"""


SCREEN_LAYER_RHOS: tuple[float, ...] = (0.1, 0.2, 0.4)
"""Three relative displacements for the layer scan: the floor, middle and ceiling of the safe band.

One strength would confound "this layer does nothing" with "this layer needs a different push", and
the whole point of the scan is to stop picking layers by an artefact. These three are the published
0.1-0.4 envelope's endpoints and midpoint, so the scan stays inside the range where a shift is
attributable to steering rather than to damage; the strength stage is what goes past both edges.
"""


# --------------------------------------------------------------------------------------
# Artifact IO
# --------------------------------------------------------------------------------------


def append_jsonl(path: Path, rows: Iterable[DataclassInstance]) -> int:
    """Append dataclass rows to a JSONL file as they are produced. Returns how many were written."""
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            written += 1
    return written


def claim_jsonl_artifact(path: Path) -> Path:
    """Return ``path`` for a stage to append to, refusing one an earlier run already wrote.

    Every stage appends its records as it produces them, so a retry into the same ``--out-dir`` puts
    a second copy of every finished cell in the same file, and nothing downstream can tell the copies
    apart: :func:`summarise_cells` groups on ``(layer, arm, position_arm, rho)`` and reports
    ``n_rows=len(group)`` with medians over whatever rows carry that key, so a stage run twice
    reports twice the sample size at the same coefficients. The key carries no fit or seed identity
    either, so re-fitting the direction and re-running into the same directory pools generations made
    under two DIFFERENT directions under one median.

    Fails closed rather than de-duplicating on read. A retry IS the anticipated workflow here -- the
    stages record ``cells_skipped_by_deadline`` and :func:`read_jsonl` tolerates a truncated final
    line -- and precisely because it is anticipated there is no key on a record that could separate
    an intended continuation from an accidental second run, so the choice is between refusing and
    guessing.

    Called before the model is loaded, so a refusal costs nothing rather than arriving after minutes
    of weight loading on a rented box.
    """
    if path.exists():
        raise FileExistsError(
            f"{path} already holds records from an earlier run of this stage, and appending would put "
            "a second copy of every finished cell in one file. The readout groups cells with no fit or "
            "seed identity in the key, so the duplicates would be reported as extra items -- or, after "
            "a re-fit, as two different directions pooled under one median. Pass a fresh --label to "
            "write beside it, or a fresh --out-dir, or delete it deliberately."
        )
    return path


def _cell_key(row: dict[str, object]) -> CellKey:
    """Type a behavioural row's (layer, position_arm, arm, rho) cell identity off its JSON."""
    return (
        int(row["layer"]),  # pyright: ignore[reportArgumentType]
        str(row["position_arm"]),
        str(row["arm"]),
        float(row["rho"]),  # pyright: ignore[reportArgumentType]
    )


def resume_jsonl_artifact(path: Path, *, rows_per_cell: int) -> tuple[Path, frozenset[CellKey]]:
    """Rewrite ``path`` keeping only cells at their full complement, and return those cells' keys.

    The resume counterpart to :func:`claim_jsonl_artifact` (owner directive, 2026-08-28: a box that
    dies early must never force a restart from zero). The claim guard refuses a blind re-run
    because no key on a record separates a continuation from an accidental double-run; this
    separates them deliberately, keyed on the full complement: a cell whose row count equals
    ``rows_per_cell`` under the CURRENT configuration is kept and skipped upstream, and every other
    row -- a partial cell from a mid-run death, a truncated final line, or a cell produced under a
    different depth -- is dropped and regenerated. Dropping loses nothing: generation seeds derive
    from (prompt_index, sample_index), never from execution order, so a resumed grid is
    bit-compatible with an unbroken one. The rewrite goes through a temp file and an atomic
    replace, so a second death mid-rewrite leaves the old file or the new one, never a torn one.
    """
    if not path.exists():
        return path, frozenset()
    rows = read_jsonl(path)
    by_cell: dict[CellKey, int] = defaultdict(int)
    for row in rows:
        by_cell[_cell_key(row)] += 1
    complete = frozenset(key for key, count in by_cell.items() if count == rows_per_cell)
    kept_rows = [row for row in rows if _cell_key(row) in complete]
    tmp = path.with_suffix(path.suffix + ".resume-tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept_rows), encoding="utf-8"
    )
    tmp.replace(path)
    logger.info(
        f"resume: kept {len(complete)} complete cells ({len(kept_rows)} rows) of {len(by_cell)} "
        f"found; dropped {len(rows) - len(kept_rows)} rows from partial or differently-shaped "
        "cells for regeneration"
    )
    return path, complete


def _claim_or_resume(
    artifact: Path, *, resume: bool, rows_per_cell: int
) -> tuple[Path, frozenset[CellKey]]:
    """Claim a fresh artifact, or -- under ``--resume`` -- keep its complete cells for skipping."""
    if resume:
        return resume_jsonl_artifact(artifact, rows_per_cell=rows_per_cell)
    return claim_jsonl_artifact(artifact), frozenset()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL artifact, tolerating a trailing partial line from an interrupted stage.

    A stage killed mid-write leaves a truncated final line. Dropping exactly that line is recovery of
    a real artifact; dropping a line anywhere else would be silent data loss, so a malformed line
    that is not the last one raises.
    """
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            logger.warning(f"dropping a truncated final line of {path} ({len(line)} chars)")
    return rows


def labelled(name: str, label: str) -> str:
    """Insert ``label`` before an artifact's extension so units sharing an out-dir do not collide.

    The behavioural read runs as several units against one fitted direction -- a wide grid, the full
    control set, then the thinking tier -- and they must share an ``--out-dir`` because that is where
    the fit artifacts live. Without a label the second unit would overwrite the first unit's
    completion marker, and a chain that lost a marker would look like a stage that never ran.
    """
    if not label:
        return name
    stem, _, suffix = name.rpartition(".")
    return f"{stem}_{label}.{suffix}" if stem else f"{name}_{label}"


def write_summary(out_dir: Path, stage: str, payload: dict[str, object], label: str = "") -> Path:
    """Write a stage's completion marker. Called last, on purpose: see the module docstring."""
    path = out_dir / labelled(f"{stage}_summary.json", label)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    logger.info(f"wrote {stage} summary to {path}")
    return path


def _deadline_from(seconds: float | None) -> float | None:
    return None if seconds is None else time.monotonic() + seconds


# --------------------------------------------------------------------------------------
# The isolation stage
# --------------------------------------------------------------------------------------


def stage_isolation(args: argparse.Namespace) -> int:
    """Run the recurrent-state isolation report, optionally under the residue-injection sabotage.

    Returns a process exit code. The ordinary run already carries its own sabotage -- the
    ``shared_cache_branching`` check deliberately shares a cache and must observe corruption -- so a
    clean verdict here means both "our path does not leak" and "this instrument can see a leak".
    Under ``--sabotage-inject-residue`` the expectation on ``steered_run_between`` INVERTS: it must
    fail, and the stage exits non-zero if it passes, because a sabotage that stays green means the
    check cannot see contamination and is therefore not a check.
    """
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    generator = torch.Generator().manual_seed(args.seed)
    # `get_text_config()` rather than `.text_config`: AutoModelForCausalLM on this VLM checkpoint
    # loads the text tower, so `model.config` is ALREADY a Qwen3_5TextConfig with no `.text_config`
    # attribute, while a wrapper config would have one. The accessor is right in both shapes.
    dim = int(model.config.get_text_config().hidden_size)  # pyright: ignore[reportAttributeAccessIssue]
    direction = torch.randn(dim, generator=generator)
    prompts = build_prompts(n=2, seed=args.seed)
    checks = cache_isolation_report(
        model,
        tokenizer,
        probe_prompt=prompts[0],
        interference_prompt=prompts[1],
        layer=args.layer,
        direction=direction,
        alpha=args.isolation_alpha,
        inject_residue=args.sabotage_inject_residue,
    )
    for check in checks:
        logger.info(f"{check.name}: {'PASS' if check.passed else 'FAIL'} -- {check.detail}")
    verdict, ok = isolation_verdict(checks, sabotage=args.sabotage_inject_residue)
    payload: dict[str, object] = {
        "stage": "isolation",
        "model_id": args.model_id,
        "layer": args.layer,
        "alpha": args.isolation_alpha,
        "sabotage_inject_residue": args.sabotage_inject_residue,
        "verdict": verdict,
        "ok": ok,
        "checks": [asdict(check) for check in checks],
    }
    (args.out_dir / labelled(ISOLATION_ARTIFACT, args.label)).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_summary(args.out_dir, "isolation", payload, args.label)
    logger.info(f"isolation verdict: {verdict}")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
# The fit stage
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitGeneration:
    """One instructed generation used to fit the direction, with its compliance score."""

    prompt_index: int
    side: str
    prompt_text: str
    response_text: str
    response_tokens: int
    hit_token_cap: bool
    target_ratio: float | None
    complied: bool


SIDE_TARGET = "target"
SIDE_BASELINE = "baseline"


@dataclass(frozen=True)
class ComplianceGate:
    """Per-side instruction compliance across the fit's two sides, and the refusal it implies.

    Pooled over both sides this gate cannot fire, which is why it is a dataclass with a pure
    constructor rather than one division in the middle of the stage. The baseline side complies by
    answering in English, which the model does unprompted at close to 100%, so with equal n per side
    the baseline alone contributes about 0.5 to a pooled rate -- at or above the default
    ``--min-compliance`` whatever the target side did. A fit in which NO target prompt followed its
    instruction therefore passed, and the direction written to ``directions.pt`` was a difference of
    means between two English answer sets: exactly the outcome the refusal message describes, with
    nothing in the artifact saying so. The threshold has to bind on the weaker side.
    """

    n_target: int
    n_target_complied: int
    target_rate: float
    n_baseline: int
    n_baseline_complied: int
    baseline_rate: float
    pooled_rate: float
    min_compliance: float
    weakest_side: str
    weakest_rate: float
    ok: bool
    failure: str | None


def compliance_gate(
    target_gens: Sequence[FitGeneration],
    baseline_gens: Sequence[FitGeneration],
    *,
    min_compliance: float,
) -> ComplianceGate:
    """Score both sides of the fit separately and refuse when the WEAKER one is below threshold.

    Pure given the generations, so the refusal can be driven red on a synthetic fit whose target side
    complied zero times without loading a model -- which is the only way this gate gets watched to
    fail. The pooled rate is still recorded, labelled as pooled, because it is what the previous
    artifacts carry and a reader comparing them needs the comparable number.

    An empty side raises rather than scoring 0.0: a side with no generations at all is a broken
    generation loop, and refusing it as non-compliance would report the wrong diagnosis.
    """
    if not target_gens or not baseline_gens:
        raise ValueError(
            f"the compliance gate needs generations on both sides, got {len(target_gens)} target "
            f"and {len(baseline_gens)} baseline: a side with none is a broken generation loop rather "
            "than a non-compliant fit"
        )
    n_target_complied = sum(gen.complied for gen in target_gens)
    n_baseline_complied = sum(gen.complied for gen in baseline_gens)
    target_rate = n_target_complied / len(target_gens)
    baseline_rate = n_baseline_complied / len(baseline_gens)
    weakest_side = SIDE_TARGET if target_rate <= baseline_rate else SIDE_BASELINE
    weakest_rate = min(target_rate, baseline_rate)
    failure = None
    if weakest_rate < min_compliance:
        failure = (
            f"the {weakest_side} side's instruction compliance {weakest_rate:.3f} is below "
            f"--min-compliance {min_compliance} (target {target_rate:.3f} over {len(target_gens)} "
            f"generations, baseline {baseline_rate:.3f} over {len(baseline_gens)}): the contrast set "
            "does not contain the contrast, so any fitted direction would be a difference of means "
            "between two answer sets that differ by nothing the instructions asked for"
        )
    return ComplianceGate(
        n_target=len(target_gens),
        n_target_complied=n_target_complied,
        target_rate=target_rate,
        n_baseline=len(baseline_gens),
        n_baseline_complied=n_baseline_complied,
        baseline_rate=baseline_rate,
        pooled_rate=(n_target_complied + n_baseline_complied)
        / (len(target_gens) + len(baseline_gens)),
        min_compliance=min_compliance,
        weakest_side=weakest_side,
        weakest_rate=weakest_rate,
        ok=failure is None,
        failure=failure,
    )


def _instructed_side(  # noqa: PLR0913 - one side is the model, the prompts, its script and the budget
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    *,
    side: str,
    score_script: str,
    want_target: bool,
    max_new_tokens: int,
) -> tuple[list[FitGeneration], list[dict[int, torch.Tensor]]]:
    """Generate one side of the symmetric instructed pair and pool its response activations.

    Compliance is scored on the side's OWN instruction: the target side complies when its response is
    mostly in the target script, the baseline side when it is mostly not. A side that did not comply
    contributes nothing but noise to a difference of means, so the rate is recorded per generation
    rather than assumed.
    """
    sampling = non_thinking_sampling(greedy=True, max_new_tokens=max_new_tokens)
    generations: list[FitGeneration] = []
    pooled: list[dict[int, torch.Tensor]] = []
    for index, prompt in enumerate(prompts):
        record = generate_response(model, tokenizer, prompt, thinking=False, sampling=sampling)
        score = script_score(record.response_text, score_script)
        ratio = score.ratio
        complied = ratio is not None and (
            (ratio > _COMPLIANCE_RATIO) if want_target else (ratio < _COMPLIANCE_RATIO)
        )
        generations.append(
            FitGeneration(
                prompt_index=index,
                side=side,
                prompt_text=prompt,
                response_text=record.response_text,
                response_tokens=record.n_generated,
                hit_token_cap=record.hit_token_cap,
                target_ratio=ratio,
                complied=complied,
            )
        )
        positionwise = capture_record_activations(model, record)
        pooled.append(pool_positions(positionwise, response_positions(record), pooling="mean"))
    return generations, pooled


def stage_fit(args: argparse.Namespace) -> int:
    """Fit a per-layer direction from symmetric instructed pairs, and measure the residual scales.

    The contrast is "Reply in Chinese." against "Reply in English." on the same questions, so the
    only surviving difference is the language rather than the presence of an instruction. Pooling is
    over the *response* positions, which is the extraction position the steering literature reports
    as the stronger one, and the arms later steer the PLAIN prompt that neither side of the fit saw.
    """
    generations_path = claim_jsonl_artifact(
        args.out_dir / labelled(FIT_GENERATIONS_ARTIFACT, args.label)
    )
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    target_prompts = build_prompts(
        n=args.n_prompts, seed=args.seed, instruction_script=args.target_script
    )
    baseline_prompts = build_prompts(
        n=args.n_prompts, seed=args.seed, instruction_script=SCRIPT_LATIN
    )
    plain_prompts = build_prompts(n=args.n_prompts, seed=args.seed)

    target_gens, target_pooled = _instructed_side(
        model,
        tokenizer,
        target_prompts,
        side="target",
        score_script=args.target_script,
        want_target=True,
        max_new_tokens=args.max_new_tokens,
    )
    baseline_gens, baseline_pooled = _instructed_side(
        model,
        tokenizer,
        baseline_prompts,
        side="baseline",
        score_script=args.target_script,
        want_target=False,
        max_new_tokens=args.max_new_tokens,
    )
    append_jsonl(generations_path, [*target_gens, *baseline_gens])

    all_gens = [*target_gens, *baseline_gens]
    gate = compliance_gate(target_gens, baseline_gens, min_compliance=args.min_compliance)
    logger.info(
        f"instruction compliance: target {gate.target_rate:.3f} over {gate.n_target}, baseline "
        f"{gate.baseline_rate:.3f} over {gate.n_baseline}; the gate binds on the {gate.weakest_side} "
        f"side at {gate.weakest_rate:.3f} against --min-compliance {gate.min_compliance}"
    )

    layers = sorted(target_pooled[0])
    positives = {layer: torch.stack([row[layer] for row in target_pooled]) for layer in layers}
    negatives = {layer: torch.stack([row[layer] for row in baseline_pooled]) for layer in layers}
    directions = {
        layer: positives[layer].mean(dim=0) - negatives[layer].mean(dim=0) for layer in layers
    }

    scales = _measure_plain_scales(model, tokenizer, plain_prompts, layers)
    payload: dict[str, object] = {
        "stage": "fit",
        "model_id": args.model_id,
        "seed": args.seed,
        "target_script": args.target_script,
        "n_prompts": args.n_prompts,
        "instructions": {
            "target": LANGUAGE_INSTRUCTIONS[args.target_script],
            "baseline": LANGUAGE_INSTRUCTIONS[SCRIPT_LATIN],
        },
        "compliance_gate": asdict(gate),
        "instruction_compliance_pooled": gate.pooled_rate,
        "min_compliance": args.min_compliance,
        "n_generations": len(all_gens),
        "layers": layers,
        "direction_norms": {str(layer): float(directions[layer].norm()) for layer in layers},
        "hit_token_cap": sum(gen.hit_token_cap for gen in all_gens),
        "response_tokens_median": statistics.median(gen.response_tokens for gen in all_gens),
    }
    if not gate.ok:
        payload["ok"] = False
        payload["failure"] = gate.failure
        write_summary(args.out_dir, "fit", payload, args.label)
        logger.error(str(gate.failure))
        return 1

    torch.save(
        {
            "directions": {layer: directions[layer] for layer in layers},
            "positives": positives,
            "negatives": negatives,
            "target_script": args.target_script,
            "model_id": args.model_id,
            "seed": args.seed,
        },
        args.out_dir / DIRECTIONS_ARTIFACT,
    )
    (args.out_dir / SCALES_ARTIFACT).write_text(
        json.dumps({str(scale.layer): asdict(scale) for scale in scales}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    continuations = [
        asdict(
            ScoredContinuation(
                prompt=plain,
                target_text=target.response_text,
                baseline_text=baseline.response_text,
            )
        )
        for plain, target, baseline in zip(plain_prompts, target_gens, baseline_gens, strict=True)
    ]
    (args.out_dir / CONTINUATIONS_ARTIFACT).write_text(
        json.dumps(continuations, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["ok"] = True
    write_summary(args.out_dir, "fit", payload, args.label)
    return 0


def _measure_plain_scales(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    layers: Sequence[int],
) -> list[ResidualScale]:
    """Per-layer residual scale on the PLAIN prompts, which are what the arms steer.

    Measured on the prompts the intervention will actually run over, not on the instructed fit
    prompts: alpha is a fraction of the residual magnitude the steering encounters, and the two
    prompt sets differ by an instruction clause.
    """
    per_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
    for prompt in prompts:
        record = generate_response(
            model,
            tokenizer,
            prompt,
            thinking=False,
            sampling=non_thinking_sampling(greedy=True, max_new_tokens=1),
        )
        positionwise = capture_record_activations(model, record)
        for layer in layers:
            per_layer[layer].append(positionwise[layer])
    return [
        residual_scale_stats(torch.cat(per_layer[layer], dim=0), layer=layer) for layer in layers
    ]


# --------------------------------------------------------------------------------------
# Loading what fit produced
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitArtifacts:
    """Everything the screening and behavioural stages need out of the ``fit`` stage."""

    directions: dict[int, torch.Tensor]
    positives: dict[int, torch.Tensor]
    negatives: dict[int, torch.Tensor]
    scales: dict[int, ResidualScale]
    continuations: list[ScoredContinuation]
    plain_prompts: list[str]
    target_script: str


def load_fit(out_dir: Path) -> FitArtifacts:
    """Load the fit stage's artifacts, refusing a partial set rather than proceeding on defaults."""
    summary_path = out_dir / "fit_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} is absent, so the fit stage never finished: its JSONL may exist while "
            "the axes do not. Run the fit stage first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("ok", False):
        raise ValueError(
            f"the fit stage recorded ok=false: {summary.get('failure', 'no reason given')}"
        )
    blob = torch.load(out_dir / DIRECTIONS_ARTIFACT, weights_only=True)
    scales_raw = json.loads((out_dir / SCALES_ARTIFACT).read_text(encoding="utf-8"))
    continuations_raw = json.loads((out_dir / CONTINUATIONS_ARTIFACT).read_text(encoding="utf-8"))
    return FitArtifacts(
        directions={int(layer): vec for layer, vec in blob["directions"].items()},
        positives={int(layer): vec for layer, vec in blob["positives"].items()},
        negatives={int(layer): vec for layer, vec in blob["negatives"].items()},
        scales={int(key): ResidualScale(**value) for key, value in scales_raw.items()},
        continuations=[ScoredContinuation(**row) for row in continuations_raw],
        plain_prompts=[row["prompt"] for row in continuations_raw],
        target_script=str(blob["target_script"]),
    )


def _screen_continuations(fit: FitArtifacts, n: int) -> list[ScoredContinuation]:
    """Return the continuations the cheap screen scores: a prefix of the fitted set.

    The published 128-pair floor is a requirement on the DIRECTION FIT -- a difference of means over too
    few pairs is dominated by sampling noise in the means. It is not a requirement on how many items the
    screen then scores: the screen's statistics are a per-item median and a per-item slope, which are
    fine at a few dozen items, and scoring all 128 at every one of a few thousand cells is where the
    wall clock goes. So the fit stays at the floor and the screen takes a prefix, with the count on the
    stage summary so a reader knows which it was.
    """
    if n < 1:
        raise ValueError(f"the screen needs at least one continuation, got {n}")
    return list(fit.continuations[:n])


def _arms_for(fit: FitArtifacts, layer: int, *, n_each: int, seed: int) -> dict[str, torch.Tensor]:
    return control_directions(
        fit.directions[layer],
        positive=fit.positives[layer],
        negative=fit.negatives[layer],
        n_each=n_each,
        seed=seed + layer,
    )


# --------------------------------------------------------------------------------------
# The two screening stages (the cheap, stated-preference tier)
# --------------------------------------------------------------------------------


def _gap_records(  # noqa: PLR0913 - one cell is the model, the fit, the axis arm and the knobs
    model: AutoModelForCausalLM,
    fit: FitArtifacts,
    *,
    layer: int,
    arm: str,
    direction: torch.Tensor,
    rho: float,
    position_arm: PositionArm,
    unsteered: GapReadout,
    encoded: EncodedScoringBatch,
) -> list[GapRecord]:
    """Score one screening cell into per-prompt records against the unsteered readout at that layer.

    Three numbers per prompt per coefficient, all in the same units and all sign-preserving: the
    unsteered gap, the steered gap, and their difference. The non-identifiability paper that motivates
    the orthogonal-component control has no unsteered baseline at all, which is exactly why its
    "orthogonal works as well as the direction" cannot be separated from "neither did anything"; not
    repeating that is cheap.
    """
    alpha = alpha_from_relative_displacement(rho, fit.scales[layer])
    readout = score_teacher_forced_gap(
        model,
        encoded,
        SteeringCell(layer=layer, direction=direction, alpha=alpha, position_arm=position_arm),
    )
    kls = kl_divergence_rows(unsteered.last_prompt_logits, readout.last_prompt_logits)
    return [
        GapRecord(
            layer=layer,
            arm=arm,
            position_arm=position_arm,
            rho=rho,
            alpha=alpha,
            prompt_index=index,
            gap=readout.gaps[index],
            gap_shift=readout.gaps[index] - unsteered.gaps[index],
            baseline_gap=unsteered.gaps[index],
            target_logprob=readout.target_logprobs[index],
            baseline_logprob=readout.baseline_logprobs[index],
            kl_last_token=kls[index],
            padded_width=readout.padded_width,
            score_batch_rows=readout.batch_rows,
            hook_calls=readout.invocations.calls,
            hook_selected_positions=readout.invocations.selected_positions,
            hook_steered_positions=readout.invocations.steered_positions,
        )
        for index in range(len(readout.gaps))
    ]


def _unsteered_readout(
    model: AutoModelForCausalLM,
    fit: FitArtifacts,
    layer: int,
    *,
    encoded: EncodedScoringBatch,
) -> GapReadout:
    """Return the alpha-zero readout: the baseline every shift and every KL is measured against.

    Run through the same hook at alpha 0 rather than with no hook at all, so the baseline shares every
    code path with the steered arms and a bug in the hook plumbing cannot show up as an effect.
    """
    readout = score_teacher_forced_gap(
        model,
        encoded,
        SteeringCell(layer=layer, direction=fit.directions[layer], alpha=0.0, position_arm="all"),
    )
    if readout.invocations.calls == 0:
        raise AssertionError(f"the alpha-zero hook never fired at layer {layer}")
    if readout.invocations.steered_positions != 0:
        raise AssertionError(
            f"the alpha-zero baseline steered {readout.invocations.steered_positions} positions at "
            f"layer {layer}; it must steer none or it is not a baseline"
        )
    if readout.invocations.selected_positions == 0:
        raise AssertionError(
            f"the alpha-zero baseline selected no positions at layer {layer}, so it did not run the "
            "same mask and hook path as the steered arms and is not a matched baseline"
        )
    return readout


def stage_screen_layers(args: argparse.Namespace) -> int:
    """Teacher-forced preference shift at every layer, real against placebo and shuffled label."""
    _refuse_position_arms_on_a_screen(
        args, "screen-layers scores the all-positions arm by construction"
    )
    fit = load_fit(args.out_dir)
    continuations = _screen_continuations(fit, args.n_screen_prompts)
    path = claim_jsonl_artifact(args.out_dir / labelled(SCREEN_LAYERS_ARTIFACT, args.label))
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    # Tokenised once for the whole stage: the encoding depends on no cell parameter, so re-deriving it
    # per cell cost a full chat-template render and tokenisation of every prompt, twice, per cell.
    encoded = encode_scoring_batch(tokenizer, continuations, batch_rows=args.score_batch_rows)
    layers = sorted(fit.directions)
    deadline = _deadline_from(args.deadline_seconds)
    written = 0
    attempted = 0
    skipped: list[str] = []
    for layer in layers:
        if deadline is not None and time.monotonic() > deadline:
            skipped.append(f"layer {layer}")
            continue
        unsteered = _unsteered_readout(model, fit, layer, encoded=encoded)
        arms = _arms_for(fit, layer, n_each=1, seed=args.seed)
        for arm, vector in arms.items():
            for rho in SCREEN_LAYER_RHOS:
                attempted += 1
                written += append_jsonl(
                    path,
                    _gap_records(
                        model,
                        fit,
                        layer=layer,
                        arm=arm,
                        direction=vector,
                        rho=rho,
                        position_arm="all",
                        unsteered=unsteered,
                        encoded=encoded,
                    ),
                )
    excesses = screen_layer_excess(args.out_dir)
    payload: dict[str, object] = {
        "stage": "screen-layers",
        "readout": "teacher_forced_logprob_gap (STATED PREFERENCE, not behaviour)",
        "model_id": args.model_id,
        "layers_planned": layers,
        "layers_skipped_by_deadline": skipped,
        "rhos": list(SCREEN_LAYER_RHOS),
        "cells_attempted": attempted,
        "records_written": written,
        "n_screen_prompts": encoded.n_rows,
        "ok": not skipped,
        "selection_statistic": (
            "median(|gap_shift|) of the real arm minus the same median over the POOLED control rows "
            "at that layer; a layer at or below zero never beat its own placebo"
        ),
        "layer_excess": [asdict(row) for row in excesses],
        "layers_below_control": [
            row.layer for row in excesses if row.excess is not None and row.excess <= 0
        ],
        "layers_without_controls": [row.layer for row in excesses if row.excess is None],
        "top_layers": top_layers_from_screen(args.out_dir, k=args.top_k),
    }
    write_summary(args.out_dir, "screen-layers", payload, args.label)
    return 0


@dataclass(frozen=True)
class LayerExcess:
    """One screened layer's real arm measured against the control band the same screen measured.

    ``excess`` is the ranking statistic: the real arm's median absolute per-prompt shift minus the
    median over every CONTROL row at that layer, pooled across the three families rather than
    maximised over them. Pooled, because ``screen-layers`` draws one vector per family
    (``n_each=1``), so a per-family floor is a single draw and the largest of three single draws is
    upward-biased noise -- it would make the floor easier to clear the noisier the controls were. The
    per-family medians are recorded beside the pooled one so a reader can see a family that behaved
    unlike the other two.

    ``excess`` is ``None`` exactly when one side is missing: a layer whose control arms never made it
    to disk has no measured placebo floor, and whether it cleared one is unknown rather than zero.
    """

    layer: int
    real_median: float | None
    control_median: float | None
    control_median_by_family: dict[str, float]
    excess: float | None
    n_real: int
    n_control: int


def screen_layer_excess(out_dir: Path) -> list[LayerExcess]:
    """Real-versus-control medians per layer, from every ``screen_layers*.jsonl`` on disk, by layer.

    CPU only, no model, and re-derived from the artifacts on every call rather than read from a value
    written down in a previous run -- which is also what makes the layer ranking it feeds testable
    and sabotage-able without a GPU.
    """
    rows: list[dict[str, object]] = []
    for path in sorted(out_dir.glob("screen_layers*.jsonl")):
        rows.extend(read_jsonl(path))
    by_family: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        family = arm_family(str(row["arm"]))
        layer = int(row["layer"])  # pyright: ignore[reportArgumentType]
        by_family[layer][family].append(abs(float(row["gap_shift"])))  # pyright: ignore[reportArgumentType]
    excesses: list[LayerExcess] = []
    for layer in sorted(by_family):
        families = by_family[layer]
        real = families.get(ARM_REAL, [])
        controls = [
            value for family, values in families.items() if family != ARM_REAL for value in values
        ]
        real_median = statistics.median(real) if real else None
        control_median = statistics.median(controls) if controls else None
        excesses.append(
            LayerExcess(
                layer=layer,
                real_median=real_median,
                control_median=control_median,
                control_median_by_family={
                    family: statistics.median(values)
                    for family, values in sorted(families.items())
                    if family != ARM_REAL
                },
                excess=(
                    None
                    if real_median is None or control_median is None
                    else real_median - control_median
                ),
                n_real=len(real),
                n_control=len(controls),
            )
        )
    return excesses


def top_layers_from_screen(out_dir: Path, *, k: int, allow_late_layers: bool = False) -> list[int]:
    """Layers whose real arm beats its own control band by the most, best first.

    This is how ``screen-strength`` and ``behaviour`` pick their layers when not told: recomputed from
    every ``screen_layers*.jsonl`` on disk each time it is asked, never read from a value written down
    in a previous run. Medians rather than means throughout, because one runaway prompt is exactly
    what made the previous causal tier's layer choice an artefact.

    **Ranked on the excess over the control band, not on the real arm alone.** A plain argmax over the
    real arm's median is the defect ``select_peak_layers`` carried until ``PEAK_LAYER_MIN_EXCESS``
    landed: the best of thirty-two layers that all did nothing is a draw from a flat distribution
    wearing the word "peak", and this screen has already measured the placebo, shuffled-label and
    orthogonal-component arms at every one of those same layers and written them to the same file. The
    excess is defined in :class:`LayerExcess`.

    A layer whose excess is at or below zero is still ranked and still returned, with a warning naming
    it and its excess: an operator who reads "nothing cleared its placebo" can decide what to do,
    whereas a stage that refused would just be a stage that died with the screen already paid for. A
    layer with NO control rows is excluded from selection instead, because its floor was never
    measured -- treating an unmeasured floor as zero is how a layer gets the behavioural budget on the
    strength of a comparison nobody made.

    Layers at or above :data:`~reward_hacking.interp.steer_validation.MAX_LAYER_FRACTION` of the depth
    are excluded from SELECTION unless ``allow_late_layers``. They are still swept and still reported;
    they are just not where the expensive behavioural units get spent, because the published failure
    there is a "high-gain perturbation trap" that collapses into repetitive loops and because a
    direction that close to the unembedding steers the output distribution rather than the computation.
    The depth is read off the artifact's own layer range rather than from a config, so the guard works
    on whatever model produced the screen.
    """
    excesses = screen_layer_excess(out_dir)
    if not excesses:
        return []
    depth = max(row.layer for row in excesses) + 1
    cutoff = depth * MAX_LAYER_FRACTION
    uncontrolled = [row.layer for row in excesses if row.excess is None]
    if uncontrolled:
        logger.warning(
            f"layers {uncontrolled} carry no control rows in the screen artifact, so no placebo floor "
            "was measured there; excluding them from selection rather than ranking them against zero"
        )
    scored = [(row.layer, row.excess) for row in excesses if row.excess is not None]
    if not scored:
        logger.warning(
            "no layer in the screen artifact carries both a real arm and a control band, so no layer "
            "can be ranked against its own placebo; returning none rather than ranking on the real "
            "arm alone"
        )
        return []
    eligible = [pair for pair in scored if allow_late_layers or pair[0] < cutoff]
    if not eligible:
        logger.warning(
            f"every screened layer with a control band sits at or above {MAX_LAYER_FRACTION} of the "
            f"{depth}-layer depth; falling back to the whole range rather than returning nothing"
        )
        eligible = scored
    ranked = sorted(eligible, key=lambda pair: pair[1], reverse=True)
    excluded = sorted({layer for layer, _ in scored} - {layer for layer, _ in eligible})
    if excluded:
        logger.info(
            f"layer selection excluded the perturbation-trap band {excluded} (cutoff {cutoff:.1f})"
        )
    best_layer, best_excess = ranked[0]
    if best_excess <= 0:
        logger.warning(
            f"no screened layer cleared its control band: the best is layer {best_layer} at an excess "
            f"of {best_excess:.4f} over the pooled placebo/shuffled/orthogonal median, so the "
            "behavioural budget is about to be spent on a layer that did not beat its own placebo"
        )
    return [layer for layer, _ in ranked[:k]]


def stage_screen_strength(args: argparse.Namespace) -> int:
    """Signed strength sweep at the chosen layers, across all three position arms and every control."""
    _refuse_position_arms_on_a_screen(args, "screen-strength always sweeps all three position arms")
    fit = load_fit(args.out_dir)
    layers = args.layers or top_layers_from_screen(args.out_dir, k=args.top_k)
    if not layers:
        raise ValueError(
            "no layers to sweep: pass --layers, or run screen-layers first so they can be selected "
            "from its artifact"
        )
    continuations = _screen_continuations(fit, args.n_screen_prompts)
    path = claim_jsonl_artifact(args.out_dir / labelled(SCREEN_STRENGTH_ARTIFACT, args.label))
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    encoded = encode_scoring_batch(tokenizer, continuations, batch_rows=args.score_batch_rows)
    scales = list(DEFAULT_RHOS)
    deadline = _deadline_from(args.deadline_seconds)
    written = 0
    attempted = 0
    skipped: list[str] = []
    for layer in layers:
        unsteered = _unsteered_readout(model, fit, layer, encoded=encoded)
        arms = _arms_for(fit, layer, n_each=args.n_control_draws, seed=args.seed)
        for position_arm in POSITION_ARMS:
            for arm, vector in arms.items():
                for rho in scales:
                    if deadline is not None and time.monotonic() > deadline:
                        skipped.append(f"L{layer}/{position_arm}/{arm}/{rho}")
                        continue
                    attempted += 1
                    written += append_jsonl(
                        path,
                        _gap_records(
                            model,
                            fit,
                            layer=layer,
                            arm=arm,
                            direction=vector,
                            rho=rho,
                            position_arm=position_arm,
                            unsteered=unsteered,
                            encoded=encoded,
                        ),
                    )
    payload: dict[str, object] = {
        "stage": "screen-strength",
        "readout": "teacher_forced_logprob_gap (STATED PREFERENCE, not behaviour)",
        "model_id": args.model_id,
        "layers": layers,
        "layer_source": "--layers" if args.layers else f"top {args.top_k} of screen_layers*.jsonl",
        "rhos": scales,
        "position_arms": list(POSITION_ARMS),
        "n_control_draws": args.n_control_draws,
        "n_screen_prompts": encoded.n_rows,
        "score_batch_rows": encoded.batch_rows,
        "cells_attempted": attempted,
        "cells_skipped_by_deadline": skipped,
        "records_written": written,
        "ok": not skipped,
    }
    write_summary(args.out_dir, "screen-strength", payload, args.label)
    return 0


def _refuse_position_arms_on_a_screen(args: argparse.Namespace, why: str) -> None:
    """Refuse ``--position-arms`` on a screening stage rather than silently ignoring it.

    A flag that parses but does nothing reads as a restriction that took effect; on a rented box
    that is a whole screen spent under a false assumption. Raised before any model load.
    """
    if args.position_arms:
        raise ValueError(f"--position-arms does not apply to the screening stages: {why}")


def planned_behaviour_cells(
    *, n_layers: int, n_position_arms: int, n_control_draws: int, n_rhos: int
) -> int:
    """Return the behavioural enumeration product: layers x position arms x arms x rhos.

    One function, so the sweep's own completeness plan, the tests that pin it, and the launch
    kit's sizing gate all name the same formula: 1 real arm plus ``len(CONTROL_ARMS)`` control
    families at ``n_control_draws`` draws each, at every (layer, position arm, rho).
    """
    return n_layers * n_position_arms * (1 + len(CONTROL_ARMS) * n_control_draws) * n_rhos


# --------------------------------------------------------------------------------------
# The behavioural stages (the read that actually counts)
# --------------------------------------------------------------------------------


def _behaviour_scales(args: argparse.Namespace) -> list[float]:
    if args.rhos:
        return list(args.rhos)
    return list(DEFAULT_RHOS)


@dataclass(frozen=True)
class BehaviourSweep:
    """What one behavioural sweep attempted, produced, skipped, and the decoding it ran under.

    Returned rather than a bare count triple so the two behaviour stages record the sampler and the
    seeding state from the same object instead of each rebuilding a description of them. Both
    summaries then carry the same keys, which is what makes a thinking-tier read comparable with the
    non-thinking one at all.
    """

    cells_planned: int
    cells_attempted: int
    records_written: int
    records_expected_per_cell: int
    cells_skipped_by_deadline: list[str]
    cells_partial: list[str]
    cells_resumed_complete: list[str]
    sampling: SamplingConfig
    generation_seed_base: int
    position_arms: tuple[PositionArm, ...]

    @property
    def is_complete(self) -> bool:
        """Every planned cell ran to full depth: what ``ok`` means, spelled as a conjunction.

        The 20260824b grid exposed the gap this closes: a deadline reaching the FINAL cell mid-way
        left that cell in neither the skip list nor any partial list, so ``ok`` keyed on the skip
        list alone could bless an incomplete artifact. Resumed-plus-attempted-plus-skipped must
        equal the independently computed plan, no cell may be partial, and the record count must
        equal the full depth of every freshly attempted cell -- each clause catches a failure the
        others can miss. ``cells_resumed_complete`` counts toward the plan because a resumed cell's
        rows are already in the artifact at full depth (that is what made it resumable); it is a
        DISTINCT ledger from the deadline skips, never conflated: a skip is missing work, a resume
        is finished work inherited from an earlier attempt at the same run.
        """
        return (
            not self.cells_skipped_by_deadline
            and not self.cells_partial
            and len(self.cells_resumed_complete)
            + self.cells_attempted
            + len(self.cells_skipped_by_deadline)
            == self.cells_planned
            and self.records_written == self.cells_attempted * self.records_expected_per_cell
        )

    def to_summary_fields(self) -> dict[str, object]:
        """Render the decoding-and-completeness block both behaviour summaries carry, spelled once.

        ``sampling_unseeded`` is the same key ``reward_hacking.train_screen`` writes, at the opposite
        value: this path DOES seed, and a reader grepping either module for the question gets an
        answer rather than silence. ``resolved_sampler`` is what reached ``generate`` as opposed to
        what was asked for -- ``SamplingConfig.seed`` is a knob transformers cannot apply, which is
        precisely why the process-global stream has to be seeded by hand.
        """
        return {
            "cells_planned": self.cells_planned,
            "cells_attempted": self.cells_attempted,
            "cells_skipped_by_deadline": list(self.cells_skipped_by_deadline),
            "cells_partial": list(self.cells_partial),
            "cells_resumed_complete": list(self.cells_resumed_complete),
            "records_written": self.records_written,
            "records_expected_per_cell": self.records_expected_per_cell,
            "position_arms": list(self.position_arms),
            "sampler": asdict(self.sampling),
            "resolved_sampler": resolved_sampler(self.sampling).as_payload(),
            "sampling_unseeded": False,
            "generation_seed_base": self.generation_seed_base,
            "generation_seed_derivation": (
                f"torch.manual_seed(generation_seed_base + {GENERATION_SEED_STRIDE} * prompt_index "
                "+ sample_index) immediately before each generation, so the real arm and every "
                "control draw the same sample per item"
            ),
            "ok": self.is_complete,
        }


def _run_behaviour(  # noqa: PLR0913 - a behavioural sweep is its model, fit, cells and sampler
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    fit: FitArtifacts,
    *,
    args: argparse.Namespace,
    layers: Sequence[int],
    scales: Sequence[float],
    thinking: bool,
    path: Path,
    resumed_cells: frozenset[CellKey],
) -> BehaviourSweep:
    """Generate and score every (layer, position arm, control arm, strength) cell, appending as it goes.

    Cell order is deliberate: for each layer the ``real`` arm runs before the controls and the
    positive strengths before the negatives, so a deadline that truncates the sweep truncates the
    least informative end of it. Every finished cell is already on disk.

    Every cell runs off the same per-item seed sequence, ``args.seed`` derived per
    ``(prompt_index, sample_index)`` and nothing else, so the real arm and its matched-norm controls
    see the same draws item for item. See
    :func:`~reward_hacking.interp.steer_validation.generation_seed`.
    """
    greedy = not thinking
    sampling = (
        thinking_sampling(max_new_tokens=args.thinking_max_new_tokens)
        if thinking
        else non_thinking_sampling(greedy=True, max_new_tokens=args.max_new_tokens)
    )
    prompts = fit.plain_prompts[: args.n_behaviour_prompts]
    position_arms: tuple[PositionArm, ...] = (
        tuple(args.position_arms) if args.position_arms else POSITION_ARMS
    )
    # Planned counts are computed INDEPENDENTLY of the loop, so ``is_complete`` compares the sweep
    # against a plan rather than against its own bookkeeping (attempted+skipped trivially equals a
    # planned count accumulated inside the loop, which would make the check tautological).
    records_expected_per_cell = len(prompts) * args.n_samples
    expected_arms = 1 + len(CONTROL_ARMS) * args.n_control_draws
    cells_planned = planned_behaviour_cells(
        n_layers=len(layers),
        n_position_arms=len(position_arms),
        n_control_draws=args.n_control_draws,
        n_rhos=len(scales),
    )
    deadline = _deadline_from(args.deadline_seconds)
    written = 0
    attempted = 0
    skipped: list[str] = []
    partial: list[str] = []
    resumed_labels: list[str] = []
    for layer in layers:
        arms = _arms_for(fit, layer, n_each=args.n_control_draws, seed=args.seed)
        if len(arms) != expected_arms:
            raise ValueError(
                f"control_directions returned {len(arms)} arms at layer {layer}, but the plan "
                f"assumes 1 real + {len(CONTROL_ARMS)} families x {args.n_control_draws} draws = "
                f"{expected_arms}; the completeness accounting would be wrong from here on"
            )
        ordered_arms = [ARM_REAL, *[name for name in arms if name != ARM_REAL]]
        for position_arm in position_arms:
            for arm in ordered_arms:
                for rho in scales:
                    label = f"L{layer}/{position_arm}/{arm}/{rho}"
                    # Before the deadline check: a resumed cell costs nothing, so an exhausted
                    # deadline must not reclassify inherited work as a skip.
                    if (layer, position_arm, arm, rho) in resumed_cells:
                        resumed_labels.append(label)
                        continue
                    if deadline is not None and time.monotonic() > deadline:
                        skipped.append(label)
                        continue
                    attempted += 1
                    records = run_behavioural_cell(
                        model,
                        tokenizer,
                        prompts,
                        layer=layer,
                        arm=arm,
                        direction=arms[arm],
                        rho=rho,
                        alpha=alpha_from_relative_displacement(rho, fit.scales[layer])
                        if rho != 0.0
                        else 0.0,
                        position_arm=position_arm,
                        thinking=thinking,
                        greedy=greedy,
                        target_script=fit.target_script,
                        n_samples=args.n_samples,
                        sampling=sampling,
                        generation_seed_base=args.seed,
                        deadline=deadline,
                        item_order_seed=args.seed + layer,
                    )
                    written += append_jsonl(path, records)
                    if len(records) < records_expected_per_cell:
                        # A deadline reaching a cell MID-generation used to vanish here: the cell
                        # was neither skipped nor complete, and ok keyed on the skip list alone.
                        partial.append(label)
                        logger.warning(
                            f"cell {label}: PARTIAL, {len(records)} of "
                            f"{records_expected_per_cell} generations"
                        )
                    else:
                        logger.info(f"cell {label}: {len(records)} generations")
    return BehaviourSweep(
        cells_planned=cells_planned,
        cells_attempted=attempted,
        records_written=written,
        records_expected_per_cell=records_expected_per_cell,
        cells_skipped_by_deadline=skipped,
        cells_partial=partial,
        cells_resumed_complete=resumed_labels,
        sampling=sampling,
        generation_seed_base=args.seed,
        position_arms=position_arms,
    )


def _behaviour_rc(sweep: BehaviourSweep) -> int:
    """Exit code for a behavioural stage: non-zero whenever the sweep is not the whole plan.

    The summary (already written by the caller -- completion marker last, as everywhere) carries
    the same verdict in its ``ok`` field; this propagates it to the process exit code, because the
    20260824b chain marked a 48-of-168-cell grid ``rc: 0`` and every downstream marker then read
    "all units rc 0" off a truncated artifact.
    """
    if sweep.is_complete:
        return 0
    logger.error(
        f"behaviour sweep INCOMPLETE: {len(sweep.cells_resumed_complete)} resumed, "
        f"{sweep.cells_attempted} attempted, "
        f"{len(sweep.cells_skipped_by_deadline)} deadline-skipped, {len(sweep.cells_partial)} "
        f"partial of {sweep.cells_planned} planned ({sweep.records_written} fresh records vs "
        f"{sweep.cells_attempted * sweep.records_expected_per_cell} expected); the summary says "
        "ok=false and this stage exits non-zero"
    )
    return 1


def stage_behaviour(args: argparse.Namespace) -> int:
    """Run the non-thinking greedy behavioural grid: what the model actually writes, per item."""
    fit = load_fit(args.out_dir)
    layers = args.layers or top_layers_from_screen(args.out_dir, k=args.top_k)
    if not layers:
        raise ValueError("no layers to run: pass --layers or run screen-layers first")
    path, resumed_cells = _claim_or_resume(
        args.out_dir / labelled(BEHAVIOUR_ARTIFACT, args.label),
        resume=args.resume,
        rows_per_cell=min(args.n_behaviour_prompts, len(fit.plain_prompts)) * args.n_samples,
    )
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    scales = _behaviour_scales(args)
    sweep = _run_behaviour(
        model,
        tokenizer,
        fit,
        args=args,
        layers=layers,
        scales=scales,
        thinking=False,
        path=path,
        resumed_cells=resumed_cells,
    )
    write_summary(
        args.out_dir,
        "behaviour",
        {
            "stage": "behaviour",
            "readout": "generated_script_ratio (BEHAVIOUR, greedy, thinking off)",
            "model_id": args.model_id,
            "layers": layers,
            "rhos": scales,
            "n_prompts": min(args.n_behaviour_prompts, len(fit.plain_prompts)),
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            **sweep.to_summary_fields(),
        },
        args.label,
    )
    return _behaviour_rc(sweep)


def stage_behaviour_thinking(args: argparse.Namespace) -> int:
    """Repeat the behavioural read with thinking on and the full token budget, at narrowed cells.

    Narrow on purpose. This is the tier where the compounding question lives -- a decode-steered arm
    writes a perturbed state into the recurrence at every one of thousands of reasoning tokens -- and
    the way to bound it is fewer cells and a deadline, never a smaller token budget. The cap stays at
    the model's own 65,536 and ``hit_token_cap`` is on every record.
    """
    fit = load_fit(args.out_dir)
    if not args.layers:
        raise ValueError(
            "--layers is required for the thinking tier: it is deliberately narrow, so the cell is "
            "chosen from the behaviour stage's readout rather than swept"
        )
    if not args.rhos:
        raise ValueError("--rhos is required for the thinking tier, for the same reason")
    path, resumed_cells = _claim_or_resume(
        args.out_dir / labelled(BEHAVIOUR_THINKING_ARTIFACT, args.label),
        resume=args.resume,
        rows_per_cell=min(args.n_behaviour_prompts, len(fit.plain_prompts)) * args.n_samples,
    )
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    sweep = _run_behaviour(
        model,
        tokenizer,
        fit,
        args=args,
        layers=args.layers,
        scales=list(args.rhos),
        thinking=True,
        path=path,
        resumed_cells=resumed_cells,
    )
    write_summary(
        args.out_dir,
        "behaviour-thinking",
        {
            "stage": "behaviour-thinking",
            "readout": "generated_script_ratio (BEHAVIOUR, sampled, thinking ON, cap 65536)",
            "model_id": args.model_id,
            "layers": args.layers,
            "rhos": list(args.rhos),
            "n_prompts": min(args.n_behaviour_prompts, len(fit.plain_prompts)),
            "n_samples": args.n_samples,
            "thinking_max_new_tokens": args.thinking_max_new_tokens,
            **sweep.to_summary_fields(),
        },
        args.label,
    )
    return _behaviour_rc(sweep)


def stage_pick_cell(args: argparse.Namespace) -> int:
    """Print ``<layer> <rho>``: the strongest HEALTHY all-positions real-arm cell of a grid.

    The launch kit's previous inline picker trusted the artifact unconditionally: both prior grids
    truncated (71/168 and 48/168 cells), the a-run's "successful" layer pick came off the truncated
    artifact anyway, and a cell with two scorable rows could win on a median over those two. This
    stage refuses (rc 1, reason logged) unless the grid's own summary says ``ok=true``, considers
    only cells present at full planned depth with at least :data:`PICK_MIN_SCORABLE_FRACTION` of
    their rows scorable, and ranks by median ratio with the cap-hit fraction as the down-ranking
    tiebreak. Cap-heavy cells are down-ranked rather than excluded on purpose: the 20260824b wide
    unit's real all-positions arm hit the token cap on 16 of 16 generations while being the one
    validated separation from every control family, so a cap-fraction exclusion would have thrown
    away the true positive. Median, not mean, and positive rho only, for the same reasons the old
    picker gave: within-concept steerability is documented bimodal, and the negative arm is a sign
    control that must not win the pick.
    """
    summary_path = args.out_dir / labelled("behaviour_summary.json", args.label)
    if not summary_path.exists():
        logger.error(f"{summary_path} is absent, so the grid unit never finished; refusing to pick")
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("ok") is not True:
        logger.error(
            f"the grid summary records ok=false ({len(summary.get('cells_skipped_by_deadline', []))} "
            f"skipped, {len(summary.get('cells_partial', []))} partial): picking from a truncated "
            "grid is how the invalidated runs chose their cells; refusing"
        )
        return 1
    rows_per_cell = int(summary["n_prompts"]) * int(summary["n_samples"])  # pyright: ignore[reportArgumentType]
    min_scorable = math.ceil(PICK_MIN_SCORABLE_FRACTION * rows_per_cell)
    rows = read_jsonl(args.out_dir / labelled(BEHAVIOUR_ARTIFACT, args.label))
    cells: dict[tuple[int, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["arm"] != ARM_REAL or row["position_arm"] != "all":
            continue
        rho = float(row["rho"])  # pyright: ignore[reportArgumentType]
        if rho <= 0:
            continue
        cells[(int(row["layer"]), rho)].append(row)  # pyright: ignore[reportArgumentType]
    candidates: list[tuple[float, float, int, int, float]] = []
    for (layer, rho), cell_rows in sorted(cells.items()):
        scorable = [
            float(row["target_ratio"])  # pyright: ignore[reportArgumentType]
            for row in cell_rows
            if row["target_ratio"] is not None
        ]
        if len(cell_rows) != rows_per_cell:
            logger.warning(
                f"cell L{layer}/{rho}: {len(cell_rows)} of {rows_per_cell} rows, not eligible"
            )
            continue
        if len(scorable) < min_scorable:
            logger.warning(
                f"cell L{layer}/{rho}: only {len(scorable)} scorable rows of {rows_per_cell} "
                f"(floor {min_scorable}), not eligible"
            )
            continue
        cap_fraction = sum(bool(row["hit_token_cap"]) for row in cell_rows) / len(cell_rows)
        candidates.append((statistics.median(scorable), -cap_fraction, len(scorable), layer, rho))
    if not candidates:
        logger.error(
            "no eligible all-positions real-arm cell at a positive rho: every candidate was "
            "missing rows or mostly unscorable; an honest no-pick beats a median over two rows"
        )
        return 1
    median, negated_cap_fraction, n_scorable, layer, rho = max(
        candidates, key=lambda cell: (cell[0], cell[1], cell[2], -cell[3], -cell[4])
    )
    logger.info(
        f"picked L{layer}/{rho}: median {median:.3f} over {n_scorable} scorable rows, "
        f"cap-hit fraction {-negated_cap_fraction:.2f}, from {len(candidates)} eligible cells"
    )
    print(f"{layer} {rho}")  # noqa: T201 - stdout IS this stage's contract; the chain captures it
    return 0


# --------------------------------------------------------------------------------------
# Readout: per item, never mean-only
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSummary:
    """One cell's distribution over its ROWS, with the sign split spelled out.

    A mean is the one statistic this cannot be reported as. Within-concept steerability is documented
    to be bimodal, with close to half of the inputs moving the wrong way on some datasets, and a
    mixture of that shape averages to about zero -- indistinguishable from no effect at all. So the
    fields are the median, the interquartile range, the counts moving each way and the raw per-row
    values, and ``anti_fraction`` is the share of rows whose sign opposes the cell's median.

    **Rows, not items, and the two differ.** A cell is ``(layer, arm, position_arm, rho)``, and the
    behavioural family writes one row per ``(prompt, sample)`` within it, so at ``--n-samples 2`` a
    16-prompt cell has 32 rows and the interquartile range spans prompt-samples rather than prompts.
    The screening family happens to write one row per prompt, so there the two coincide. The fields
    are named ``n_rows`` and ``per_row`` because an ``n_items`` that silently meant either would be
    read as the second: :class:`ItemSteerability` is the genuinely per-item view, and
    :func:`stage_report` reports its own ``n_items`` beside these, which is a count of prompts.
    """

    layer: int
    arm: str
    arm_family: str
    position_arm: str
    rho: float
    measure: str
    n_rows: int
    n_scored: int
    n_unscorable: int
    median: float | None
    q1: float | None
    q3: float | None
    n_positive: int
    n_zero: int
    n_negative: int
    anti_fraction: float | None
    per_row: list[float | None]
    median_self_nll: float | None
    cap_hit_rate: float | None
    median_distinct_trigram: float | None


def _quartiles(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    ordered = sorted(values)
    median = statistics.median(ordered)
    if len(ordered) < _QUARTILE_MIN_N:
        return median, ordered[0], ordered[-1]
    quantiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return median, quantiles[0], quantiles[2]


def _optional_median(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def _optional_rate(flags: Sequence[object]) -> float | None:
    """Share of the rows carrying a true flag, or ``None`` when no row carries the field at all.

    The sibling of :func:`_optional_median` and for the same reason. ``summarise_cells`` serves two
    record families and only the behavioural one declares ``hit_token_cap``: a screening record has
    no such field, teacher-forced scoring has no token cap to hit, and dividing a sum of absent
    flags by the row count printed a hard ``0.00`` cap-hit rate for a metric that does not apply.
    ``None`` says "not measured here", which is a different statement from "measured, and none hit".
    """
    present = [flag for flag in flags if flag is not None]
    return (sum(bool(flag) for flag in present) / len(present)) if present else None


def summarise_cells(
    rows: Sequence[dict[str, object]], *, value_key: str, measure: str
) -> list[CellSummary]:
    """Group JSONL rows into per-cell summaries carrying every per-item value.

    Works on either artifact family: ``value_key="target_ratio"`` reads the behavioural JSONL and
    ``value_key="gap_shift"`` the screening JSONL, so one readout serves both tiers and cannot
    describe them with two different statistics. ``None`` values are counted as unscorable rather
    than coerced to zero, because a response with no letters and an English response are different
    outcomes and merging them would read a collapse as a null.
    """
    grouped: dict[tuple[int, str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["layer"]),  # pyright: ignore[reportArgumentType]
            str(row["arm"]),
            str(row["position_arm"]),
            float(row["rho"]),  # pyright: ignore[reportArgumentType]
        )
        grouped[key].append(row)
    summaries: list[CellSummary] = []
    for (layer, arm, position_arm, rho), group in sorted(grouped.items()):
        raw = [row.get(value_key) for row in group]
        per_row = [None if value is None else float(value) for value in raw]  # pyright: ignore[reportArgumentType]
        scored = [value for value in per_row if value is not None]
        median, q1, q3 = _quartiles(scored)
        positive = sum(1 for value in scored if value > 0)
        negative = sum(1 for value in scored if value < 0)
        anti: float | None = None
        if scored and median is not None and median != 0:
            anti = sum(1 for value in scored if value * median < 0) / len(scored)
        summaries.append(
            CellSummary(
                layer=layer,
                arm=arm,
                arm_family=arm_family(arm),
                position_arm=position_arm,
                rho=rho,
                measure=measure,
                n_rows=len(group),
                n_scored=len(scored),
                n_unscorable=len(group) - len(scored),
                median=median,
                q1=q1,
                q3=q3,
                n_positive=positive,
                n_zero=len(scored) - positive - negative,
                n_negative=negative,
                anti_fraction=anti,
                per_row=per_row,
                median_self_nll=_optional_median(
                    [row.get("self_nll") for row in group]  # pyright: ignore[reportArgumentType]
                ),
                cap_hit_rate=_optional_rate([row.get("hit_token_cap") for row in group]),
                median_distinct_trigram=_optional_median(
                    [row.get("distinct_trigram_ratio") for row in group]  # pyright: ignore[reportArgumentType]
                ),
            )
        )
    return summaries


@dataclass(frozen=True)
class ItemSteerability:
    """One item's dose-response across a signed coefficient grid, reduced to interpretable scalars.

    ``slope`` is the statistic Tan et al. use (arXiv:2407.12404): an ordinary-least-squares line fitted
    to the observable against the coefficient across the grid, intercept discarded, one scalar per
    input. ``anti_steerable`` is a negative slope, with no magnitude threshold -- their definition.
    ``delta_at_max`` is the simpler anchor Braun et al. (arXiv:2505.22637) prefer: the observable at the
    largest positive coefficient minus the observable unsteered, so the sign is read against a genuine
    baseline rather than against a fitted line. ``spearman`` is the monotonicity check, which is what
    turns "this item moved the wrong way" into a claim rather than a coincidence: the sign-inversion
    literature requires a monotone response across a sweep before calling an inversion an inversion.
    """

    prompt_index: int
    n_points: int
    slope: float | None
    anti_steerable: bool | None
    delta_at_max: float | None
    spearman: float | None


_OLS_MIN_N = 2
"""A slope needs two points; below that the design is degenerate rather than merely noisy."""


def _ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Least-squares slope of ``ys`` on ``xs``, or ``None`` when the design is degenerate."""
    n = len(xs)
    if n < _OLS_MIN_N:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator


def average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks with ties sharing their average, so ties do not manufacture an ordering the data lacks.

    Public and shared: :func:`games.interp_trajectory.trajectory_correlation` ranks its checkpoint
    series with this too. It used to carry its own copy, byte-identical in behaviour and independently
    spelled, which is two places for the tie convention to drift apart. The midrank convention is
    pinned by test (``[10, 20, 20, 30]`` -> ``[1.0, 2.5, 2.5, 4.0]``) rather than left implicit,
    because it is the part a reimplementation gets wrong.

    Deliberately not ``statistics.correlation(method="ranked")``: its tie handling is not the
    documented part of that API, and switching would move Spearman values already reported in
    artifacts for no gain.
    """
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2.0 + 1.0
        for index in order[position : end + 1]:
            ranks[index] = shared
        position = end + 1
    return ranks


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or ``None`` when either series is constant and has no correlation.

    ``None`` rather than 0.0, which would read as "measured, and unrelated". Shared with
    :mod:`games.interp_trajectory` for the same reason as :func:`average_ranks`; the two former
    copies were checked to agree exactly on 45 series including ties, constants and short ones before
    they were collapsed into this one.
    """
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    x_dev = [x - x_mean for x in xs]
    y_dev = [y - y_mean for y in ys]
    denominator = (sum(d * d for d in x_dev) ** 0.5) * (sum(d * d for d in y_dev) ** 0.5)
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(x_dev, y_dev, strict=True)) / denominator


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation, or ``None`` when it is undefined."""
    if len(xs) < _SPEARMAN_MIN_N:
        return None
    return pearson_correlation(average_ranks(xs), average_ranks(ys))


def item_steerability(
    rows: Sequence[dict[str, object]], *, value_key: str
) -> list[ItemSteerability]:
    """Per-item dose-response over whatever signed grid the rows cover, one record per prompt.

    Grouped by prompt across coefficients, which is the transpose of the per-cell view: a cell summary
    answers "what did this coefficient do on average", and this answers "what did the sweep do to this
    item". Anti-steerability is only visible in the second, and averaging the first is how a mixture
    with nearly half its mass pointing the wrong way reads as no effect at all.
    """
    by_item: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        by_item[int(row["prompt_index"])].append((float(row["rho"]), float(value)))  # pyright: ignore[reportArgumentType]
    records: list[ItemSteerability] = []
    for prompt_index, points in sorted(by_item.items()):
        merged: dict[float, list[float]] = defaultdict(list)
        for rho, value in points:
            merged[rho].append(value)
        rhos = sorted(merged)
        means = [statistics.fmean(merged[rho]) for rho in rhos]
        slope = _ols_slope(rhos, means)
        zero = merged.get(0.0)
        top = max((rho for rho in rhos if rho > 0), default=None)
        delta = (
            statistics.fmean(merged[top]) - statistics.fmean(zero)
            if top is not None and zero
            else None
        )
        records.append(
            ItemSteerability(
                prompt_index=prompt_index,
                n_points=len(rhos),
                slope=slope,
                anti_steerable=None if slope is None else slope < 0,
                delta_at_max=delta,
                spearman=_spearman(rhos, means),
            )
        )
    return records


def anti_steerable_fraction(items: Sequence[ItemSteerability]) -> float | None:
    """Share of items whose per-item slope points the wrong way, or ``None`` if none is defined.

    The second panel of the reporting shape worth copying: a per-item distribution is hard to read
    without one number saying how much of it opposes the whole. Published work finds datasets where
    this approaches one half, which averages to nothing.
    """
    defined = [item.anti_steerable for item in items if item.anti_steerable is not None]
    return (sum(defined) / len(defined)) if defined else None


def _fmt(value: float | None) -> str:
    return "   --  " if value is None else f"{value:7.3f}"


def format_cell_table(summaries: Sequence[CellSummary]) -> str:
    """Render per-cell summaries: median, interquartile range, sign split, coherence, cap rate."""
    header = (
        f"{'layer':>5} {'position':>9} {'scale':>7} {'arm':<26} {'n':>3} "
        f"{'median':>7} {'q1':>7} {'q3':>7} {'+/0/-':>10} {'anti':>7} {'nll':>7} {'cap':>7} {'d3':>7}"
    )
    lines = [header, "-" * len(header)]
    for cell in summaries:
        split = f"{cell.n_positive}/{cell.n_zero}/{cell.n_negative}"
        lines.append(
            f"{cell.layer:>5} {cell.position_arm:>9} {cell.rho:>7.3f} {cell.arm:<26} "
            f"{cell.n_scored:>3} {_fmt(cell.median)} {_fmt(cell.q1)} {_fmt(cell.q3)} "
            f"{split:>10} {_fmt(cell.anti_fraction)} {_fmt(cell.median_self_nll)} "
            f"{_fmt(cell.cap_hit_rate)} {_fmt(cell.median_distinct_trigram)}"
        )
    return "\n".join(lines)


REPORT_GLOBS: tuple[tuple[str, str, str], ...] = (
    ("screen_layers*.jsonl", "gap_shift", "teacher_forced_gap_shift (STATED PREFERENCE)"),
    ("screen_strength*.jsonl", "gap_shift", "teacher_forced_gap_shift (STATED PREFERENCE)"),
    ("behaviour_thinking*.jsonl", "target_ratio", "script_ratio (BEHAVIOUR, thinking on)"),
    ("behaviour*.jsonl", "target_ratio", "script_ratio (BEHAVIOUR, thinking off)"),
)
"""Glob patterns, not fixed names, so every labelled unit's artifact is picked up automatically.

A readout that names its inputs goes stale the moment a unit is added; one that globs reports
whatever is on disk. The thinking pattern is listed BEFORE the general behaviour pattern because
``behaviour*.jsonl`` would otherwise swallow it and describe sampled thinking-mode generations under
the greedy label.
"""


def stage_report(args: argparse.Namespace) -> int:
    """Summarise every artifact present in the out-dir. CPU only, no model, re-runnable.

    Computed from the JSONL on disk every time, never from a number written into a document: a
    readout that does not recompute goes stale the moment another cell lands.

    Writes its own ``report_summary.json`` last, like every other stage. Without it this stage was
    the one exception to the module's completion-marker invariant, so a finished report and a report
    that never ran looked the same from disk, and a half-written ``report.json`` -- the stage killed
    between globbing and writing -- could not be told from a complete one. The summary is
    counts-only: which source files were read and how many records and cells each yielded, which
    glob patterns matched nothing, and which isolation files were folded in. Unlabelled, because
    :data:`REPORT_ARTIFACT` is unlabelled too and the readout deliberately describes the whole
    out-dir rather than one unit of it.
    """
    report: dict[str, object] = {"stage": "report", "out_dir": str(args.out_dir), "sources": {}}
    sources: dict[str, object] = report["sources"]  # pyright: ignore[reportAssignmentType]
    seen: set[Path] = set()
    counts: dict[str, dict[str, int]] = {}
    unmatched: list[str] = []
    for pattern, value_key, measure in REPORT_GLOBS:
        matched = [path for path in sorted(args.out_dir.glob(pattern)) if path not in seen]
        if not matched:
            unmatched.append(pattern)
        for path in matched:
            seen.add(path)
            rows = read_jsonl(path)
            if not rows:
                sources[path.name] = {"records": 0, "note": "present but empty"}
                counts[path.name] = {"records": 0, "cells": 0}
                continue
            summaries = summarise_cells(rows, value_key=value_key, measure=measure)
            # Per-item dose-response is computed within each (layer, position arm, control arm) group,
            # because a slope fitted across cells that differ in more than the coefficient would be
            # measuring the layer as much as the item.
            steerability: dict[str, object] = {}
            grouped_by_arm: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
            for row in rows:
                grouped_by_arm[
                    (int(row["layer"]), str(row["position_arm"]), str(row["arm"]))  # pyright: ignore[reportArgumentType]
                ].append(row)
            for (layer, position_arm, arm), group in sorted(grouped_by_arm.items()):
                items = item_steerability(group, value_key=value_key)
                if not items:
                    continue
                steerability[f"L{layer}/{position_arm}/{arm}"] = {
                    "n_items": len(items),
                    "anti_steerable_fraction": anti_steerable_fraction(items),
                    "median_slope": _optional_median([item.slope for item in items]),
                    "median_delta_at_max": _optional_median([item.delta_at_max for item in items]),
                    "median_spearman": _optional_median([item.spearman for item in items]),
                    "per_item": [asdict(item) for item in items],
                }
            sources[path.name] = {
                "records": len(rows),
                "cells": len(summaries),
                "measure": measure,
                "summaries": [asdict(cell) for cell in summaries],
                "item_steerability": steerability,
            }
            counts[path.name] = {"records": len(rows), "cells": len(summaries)}
            print(f"\n=== {path.name} -- {measure} ({len(rows)} records, {len(summaries)} cells)")  # noqa: T201  # Intentional CLI table output.
            print(format_cell_table(summaries))  # noqa: T201  # Intentional CLI table output.
    # Globbed, so the sabotage arm's own report lands beside the real one instead of replacing it:
    # a run whose sabotage arm is missing from the readout cannot be told from one that never ran it.
    isolation_files = sorted(args.out_dir.glob("isolation_checks*.json"))
    report["isolation"] = {
        path.name: json.loads(path.read_text(encoding="utf-8")) for path in isolation_files
    }
    (args.out_dir / REPORT_ARTIFACT).write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    logger.info(f"wrote {args.out_dir / REPORT_ARTIFACT}")
    write_summary(
        args.out_dir,
        "report",
        {
            "stage": "report",
            "out_dir": str(args.out_dir),
            "artifact": REPORT_ARTIFACT,
            "sources": counts,
            "patterns_matching_nothing": unmatched,
            "isolation_files": [path.name for path in isolation_files],
            "records_total": sum(entry["records"] for entry in counts.values()),
            "cells_total": sum(entry["cells"] for entry in counts.values()),
            "ok": True,
        },
    )
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

STAGES = {
    "isolation": stage_isolation,
    "fit": stage_fit,
    "screen-layers": stage_screen_layers,
    "screen-strength": stage_screen_strength,
    "behaviour": stage_behaviour,
    "behaviour-thinking": stage_behaviour_thinking,
    "pick-cell": stage_pick_cell,
    "report": stage_report,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="corpus order, control draws, and the per-item base for the decoding seed the "
        "behavioural stages set on the process-global torch stream",
    )
    parser.add_argument("--target-script", default=SCRIPT_HAN, choices=sorted(TARGET_SCRIPTS))
    parser.add_argument(
        "--n-prompts", type=int, default=16, help="prompts used to fit the direction"
    )
    parser.add_argument(
        "--n-behaviour-prompts", type=int, default=16, help="prompts per behavioural cell"
    )
    parser.add_argument("--n-samples", type=int, default=1, help="generations per prompt per cell")
    parser.add_argument(
        "--n-control-draws", type=int, default=3, help="draws of each control family"
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument(
        "--top-k", type=int, default=3, help="layers to carry forward from the screen"
    )
    parser.add_argument("--rhos", type=float, nargs="+", default=None)
    parser.add_argument(
        "--position-arms",
        nargs="+",
        choices=list(POSITION_ARMS),
        default=None,
        help="restrict the behavioural stages to these position arms; default is all three. The "
        "screening stages take no restriction and refuse the flag: screen-strength always sweeps "
        "all three, and screen-layers scores the all-positions arm by construction. A targeted "
        "grid over the arm a pick keys on needs this, because an unrestricted grid spends two "
        "thirds of its cells on arms the pick never reads",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048, help="non-thinking answer budget"
    )
    parser.add_argument(
        "--thinking-max-new-tokens",
        type=int,
        default=65536,
        help="the model's own budget; bound runtime with --deadline-seconds and fewer cells instead",
    )
    parser.add_argument(
        "--label",
        default="",
        help="suffix for this unit's artifacts, so several units can share one --out-dir",
    )
    parser.add_argument("--deadline-seconds", type=float, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="behavioural stages only: keep an existing artifact's complete cells (counted as "
        "cells_resumed_complete, distinct from deadline skips) and regenerate every other cell; "
        "without it an existing artifact refuses, as before. Seeds derive from indices, so a "
        "resumed sweep is bit-compatible with an unbroken one",
    )
    parser.add_argument(
        "--n-screen-prompts",
        type=int,
        default=32,
        help="continuations the cheap screen scores; the 128 floor is on the FIT, not on scoring",
    )
    parser.add_argument(
        "--score-batch-rows",
        type=int,
        default=DEFAULT_SCORE_BATCH_ROWS,
        help="rows per teacher-forced forward; the LM head over a 248k vocabulary is the memory wall",
    )
    parser.add_argument("--min-compliance", type=float, default=0.5)
    parser.add_argument("--isolation-alpha", type=float, default=8.0)
    parser.add_argument("--layer", type=int, default=None, help="layer for the isolation stage")
    parser.add_argument(
        "--sabotage-inject-residue",
        action="store_true",
        help="inject a residual perturbation on the probe's second read, standing in for state left by "
        "the previous condition; the contamination check MUST then fail, and this stage exits "
        "non-zero if it does not",
    )
    args = parser.parse_args(argv)
    # A repeated value re-runs identical cells at identical seeds, doubling the reported sample
    # size while cells_planned doubles in lockstep (so is_complete stays green) -- and a resume
    # would then inherit the duplicate as already-complete, corrupting the readout a second way.
    # Refused here, before any model load.
    for flag, values in (
        ("--layers", args.layers),
        ("--rhos", args.rhos),
        ("--position-arms", args.position_arms),
    ):
        if values and len(values) != len(set(values)):
            parser.error(f"{flag} has repeated values: {values}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    return args


def main(argv: list[str] | None = None) -> int:
    """Run one stage and return its exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    if args.layer is None:
        args.layer = 12
    logger.info(f"stage={args.stage} model={args.model_id} out_dir={args.out_dir} seed={args.seed}")
    return STAGES[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
