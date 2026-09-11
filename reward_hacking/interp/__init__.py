"""Cheap-to-rigorous interpretability probes for the reward-hacking / eval-awareness leads.

The question this package exists to answer: does Qwen3.5-4B carry a "reward-hacking" (shortcut)
direction distinct from a "deception" direction, and does it register a rigged grader -- and does it
do so more strongly WHILE REASONING about the grader than while first reading it? The methods run
cheapest-first but rigor-led:

* ``directions`` / ``linear_probe`` -- diff-of-means axes and trained probes over pooled residuals,
  per-layer cosines, with a shuffled-label null and a matched-norm placebo. The correlational
  baseline.
* ``eval_awareness_probe`` -- validates the eval-awareness axis (per pooling) against both controls
  before anything leans on it.
* ``prompt_contrast`` -- projects conflicting-vs-original agentic-prompt activations (prompt-read
  time) onto the concept axes, with a generic-contradiction control to tell "rigged-test awareness"
  from "contradictory grader text".
* ``generation_capture`` -- the reasoning-phase read: capture residuals over the model's OWN
  generated tokens, with per-position bookkeeping.
* ``steering`` -- the causal side: steering, ablation, and activation patching over the residual
  stream, with the mandatory matched-norm placebo arm.
* ``jacobian`` -- the primary, most-faithful method: transport a direction through a Jacobian lens
  and decode it to a token list (owner's rule: lead with Jacobian-space, not the easiest method).

Nothing here loads a model at import time. The pure-math cores run on synthetic activations in the
offline tests; the model-capture, generation, steering, and Jacobian paths are behind functions the
CLIs call. ``prompt_contrast`` and ``eval_awareness_probe`` pull in the harness to materialise
stimuli, so they are imported from their modules directly rather than re-exported here.
"""

from reward_hacking.interp.directions import (
    LayerComparison,
    LayerOutput,
    capture_layer_output_during,
    capture_pooled_activations,
    capture_pooled_activations_multi,
    capture_positionwise_activations,
    compare_directions,
    concept_directions,
    cosine,
    diff_of_means,
    last_token_pool,
    matched_norm_random_direction,
    mean_pool,
    unit,
)
from reward_hacking.interp.generation_capture import (
    GenerationRecord,
    HFResponseGenerator,
    ResponseGenerator,
    VLLMResponseGenerator,
    capture_generation_activations,
    capture_record_activations,
    capture_record_pooled,
    capture_response_pooled,
    default_gen_engine,
    generate_response,
    matched_window_positions,
    pool_positions,
    record_from_token_ids,
    response_positions,
)
from reward_hacking.interp.jacobian import (
    JacobianConfig,
    TokenReadout,
    cap_fit_prompts,
    decode_topk,
    load_lens,
    load_model_and_lens,
    transport_and_decode,
)
from reward_hacking.interp.linear_probe import (
    CaptureSpec,
    ConceptActivations,
    ConceptPairRead,
    ConceptProbeRead,
    LayerContext,
    LayerReads,
    LogisticFit,
    ProbeConfig,
    RegularizationSweepConcept,
    RegularizationSweepPair,
    cross_validated_accuracy,
    fit_logistic_probe,
    grouped_test_masks,
    permuted_label_accuracies,
    probe_layer,
    run_linear_probe,
    standardizing_stats,
)
from reward_hacking.interp.steering import (
    PatchPrefix,
    PatchResult,
    ablate_residual,
    ablation_hook,
    activation_patch_hook,
    capture_patch_prefix,
    logit_recovery,
    residual_intervention,
    run_activation_patch,
    run_steered_generation,
    steer_residual,
    steering_directions,
    steering_hook,
)
from reward_hacking.interp.stimuli import (
    CONCEPTS,
    CONTRADICTION_PAIRS,
    DECEPTION_PAIRS,
    EVAL_AWARENESS_PAIRS,
    SHORTCUT_PAIRS,
    ContrastivePair,
    negatives,
    positives,
)

__all__ = [
    "CONCEPTS",
    "CONTRADICTION_PAIRS",
    "DECEPTION_PAIRS",
    "EVAL_AWARENESS_PAIRS",
    "SHORTCUT_PAIRS",
    "CaptureSpec",
    "ConceptActivations",
    "ConceptPairRead",
    "ConceptProbeRead",
    "ContrastivePair",
    "GenerationRecord",
    "HFResponseGenerator",
    "JacobianConfig",
    "LayerComparison",
    "LayerContext",
    "LayerOutput",
    "LayerReads",
    "LogisticFit",
    "PatchPrefix",
    "PatchResult",
    "ProbeConfig",
    "RegularizationSweepConcept",
    "RegularizationSweepPair",
    "ResponseGenerator",
    "TokenReadout",
    "VLLMResponseGenerator",
    "ablate_residual",
    "ablation_hook",
    "activation_patch_hook",
    "cap_fit_prompts",
    "capture_generation_activations",
    "capture_layer_output_during",
    "capture_patch_prefix",
    "capture_pooled_activations",
    "capture_pooled_activations_multi",
    "capture_positionwise_activations",
    "capture_record_activations",
    "capture_record_pooled",
    "capture_response_pooled",
    "compare_directions",
    "concept_directions",
    "cosine",
    "cross_validated_accuracy",
    "decode_topk",
    "default_gen_engine",
    "diff_of_means",
    "fit_logistic_probe",
    "generate_response",
    "grouped_test_masks",
    "last_token_pool",
    "load_lens",
    "load_model_and_lens",
    "logit_recovery",
    "matched_norm_random_direction",
    "matched_window_positions",
    "mean_pool",
    "negatives",
    "permuted_label_accuracies",
    "pool_positions",
    "positives",
    "probe_layer",
    "record_from_token_ids",
    "residual_intervention",
    "response_positions",
    "run_activation_patch",
    "run_linear_probe",
    "run_steered_generation",
    "standardizing_stats",
    "steer_residual",
    "steering_directions",
    "steering_hook",
    "transport_and_decode",
    "unit",
]
