"""Stage-A re-analysis of the TMAX artifacts, inference-only (no training run).

TMAX (arXiv:2606.23321) trained Qwen3.5-9B with one RL recipe on seven terminal-agent suites and
released the matched base checkpoint, the flagship arm's intermediate checkpoints and training
rollouts. This package makes three inference-only re-analyses runnable:

* :mod:`reward_hacking.tmax.artifacts` -- the registry of released artifacts (base checkpoint, the
  seven suite-named RL'd models, the flagship step-branch ladder, the rollout archive, datasets),
  keyed off verified Hugging Face ids with a fail-loud guard on anything unverified.
* :mod:`reward_hacking.tmax.rollout_analysis` -- experiment 3, runnable now with no GPU: parse
  rollout records into the deliberate-hack detector's trace schema and aggregate a base-vs-RL,
  per-suite hack-rate table.
* :mod:`reward_hacking.tmax.geometry` -- experiments 1 & 2 scaffold (GPU-deferred): direction
  geometry across checkpoints and suites, with the heavy capture injected so the module's pure
  logic is testable without a GPU.
* :mod:`reward_hacking.tmax.download` -- a snapshot-download CLI (with ``--dry-run``) for the above.
* :mod:`reward_hacking.tmax.rollout_transcripts` -- decodes the released rollout rows into turns,
  derives commands with real exit statuses, and runs the per-step scan (records, transcripts, table).
* :mod:`reward_hacking.tmax.rollout_gaming_judge` -- the reward-blind LLM judge over a stratified
  sample of those transcripts on the Bedrock batch path, with the detector-agreement and hand-read
  tooling.

``artifacts`` and ``geometry`` are re-exported here because both are torch-free at import.
``rollout_analysis``, ``rollout_transcripts`` and ``rollout_gaming_judge`` are intentionally NOT
re-exported: they import the detector package (which pulls in torch), and keeping them out of this
``__init__`` is what lets ``import reward_hacking.tmax`` and ``import reward_hacking.tmax.geometry``
stay torch-free. Import them directly when you need them:
``from reward_hacking.tmax.rollout_analysis import aggregate_hack_rates``.
"""

from reward_hacking.tmax.artifacts import (
    BASE_CHECKPOINT,
    BASE_CHECKPOINTS,
    FLAGSHIP_ROLLOUTS,
    SUITES,
    UNVERIFIED_ARTIFACTS,
    Checkpoint,
    CheckpointStage,
    EnvironmentSuite,
    HFDataset,
    RolloutArchive,
    TmaxArtifactError,
    dose_response_ladder,
    rl_models,
    stage_of,
    suite,
)
from reward_hacking.tmax.geometry import (
    CheckpointSeparability,
    DoseResponseTable,
    SuiteComparisonTable,
    SuiteLooseness,
    SuiteSeparabilityRow,
    rank_suites_by_looseness,
    run_dose_response,
    run_suite_comparison,
    summarize_direction_comparisons,
)

__all__ = [
    "BASE_CHECKPOINT",
    "BASE_CHECKPOINTS",
    "FLAGSHIP_ROLLOUTS",
    "SUITES",
    "UNVERIFIED_ARTIFACTS",
    "Checkpoint",
    "CheckpointSeparability",
    "CheckpointStage",
    "DoseResponseTable",
    "EnvironmentSuite",
    "HFDataset",
    "RolloutArchive",
    "SuiteComparisonTable",
    "SuiteLooseness",
    "SuiteSeparabilityRow",
    "TmaxArtifactError",
    "dose_response_ladder",
    "rank_suites_by_looseness",
    "rl_models",
    "run_dose_response",
    "run_suite_comparison",
    "stage_of",
    "suite",
    "summarize_direction_comparisons",
]
