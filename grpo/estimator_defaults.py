"""Repo-wide GRPO estimator defaults, with the reasoning that picked them.

One source of truth for the estimator knobs every GRPO harness here constructs
(`games/train.py`, `grpo/rlvr_math.py`, `grpo/throughput.py`), so the three sites cannot drift
apart the way they had (throughput briefly ran `scale_rewards="group"` while the others ran
`"batch"`). Deliberately import-light: `games/rewards.py` already restates a constant to stay off
the torch-heavy `grpo.throughput` import path, and this module exists so nothing has to do that
again. Audit trail: docs/scratch/opt-review-2026-08-20/grpo-final.md (gitignored, local).

Why these values, in brief:

- **Our runs are on-policy at the parameter level**, so most of TRL's variant menu is inert. With
  ``num_iterations=1`` and aligned generation/optimizer steps (both TRL defaults, neither
  overridden here), the sampling weights equal the grading weights: the policy ratio is
  identically 1, PPO clipping never binds, and grpo/dapo/dr_grpo/bnpo differ ONLY in how token
  losses are aggregated. The live choices are aggregation and advantage scaling; everything else
  (cispo, sapo, gspo, luspo, vespo, epsilon_high) is a relabel until real off-policyness lands.

- **``loss_type="dr_grpo"``, not ``"dapo"``, because of how the installed Liger path executes.**
  TRL 1.10's ``compute_liger_loss`` never forwards ``inputs["num_items_in_batch"]`` to the fused
  loss (trl/trainer/grpo_trainer.py:2956-2966, vs the non-Liger branch at :3211 which uses it), so
  Liger's DAPO normalizer falls back to the *current micro-batch's* token count
  (liger_kernel/chunked_loss/fused_linear_ppo.py:465, ``normalizer = attention_mask.sum()``). At
  micro-batch size 1 -- the production 32k-completion shape -- "dapo" therefore executes as
  original GRPO per-sequence normalization, the exact length bias DAPO exists to remove.
  ``dr_grpo``'s normalizer (batch rows x ``max_completion_length``,
  liger_kernel/chunked_loss/grpo_loss.py:236) is constant per sequence, distributes exactly across
  gradient accumulation, and matches the non-Liger branch -- the one loss whose executed semantics
  equal its paper semantics with Liger on or off. If TRL ever passes ``num_items_in_batch``
  through, "dapo" becomes an equally faithful choice; until then it is a misdescription.

- **``scale_rewards="none"``, never ``"group"``, and not ``"batch"``.** Group-mix grading gives a
  group a two-point reward distribution, and dividing by the group std cancels the payoff gap
  exactly: measured on the twin-pd-group 2B artifacts, temptation-2 and temptation-10 groups at
  the same action mix produce group-scaled advantage gaps identical to four decimals while their
  unscaled gaps differ by the designed payoff structure (up to 1.8x). Group scaling would erase,
  at the gradient level, precisely the payoff manipulations these arms exist to study. "batch"
  preserves within-batch ratios but makes the divisor a random variable of batch composition:
  measured, sigma_batch correlates 0.94-0.97 with the step's parse-failure count (the format
  channel modulates every game gradient), and falls 20-38% over a 70-step run (advantages inflate
  as the policy purifies -- the wrong direction). Our payoffs are already designed onto [0, 1]
  (games/payoffs.py), so the reward scale is the recorded, deliberate place where scale lives.

- **Effective-LR consequence, true of the gradient and not of the behaviour:** against the trained
  2B arms, "none" shrinks advantages by the sigma_batch that "batch" divided out (measured
  0.20-0.56, typically ~0.4), and dr_grpo's constant normalizer is mean-length/max-length
  (~0.33-0.39 at the observed 11-13k of a 32k budget) of the per-sequence one. Together roughly
  5-10x less gradient per step at equal ``learning_rate``, and the 2026-08-22 estimator A/B on the
  2B twin-pd-group arm (recorded beside the audit trail) measured exactly that: grad_norm 7x
  smaller early and 12x late. But the behaviour barely moved with it -- the cooperation trajectory
  came out only 1.26x shallower -- because AdamW divides the first moment by the root second
  moment and a uniform gradient rescale cancels out of that ratio. So no learning-rate
  compensation is owed for the estimator switch; the ~3e-5 figure an earlier version of this note
  named is retired. The caveat is Adam's ``adam_epsilon``: the invariance holds only while the
  root second moment sits above epsilon, and at the 9B production estimator it does not for the
  LoRA A matrices (99.3% of their entries below the default 1e-8 in the banked self arm's step-70
  optimizer state, damping the update to a median 0.108 of the nominal step; the B matrices are
  fine at 0.939). There the estimator's small gradients DO bite, through the one channel Adam
  does not normalise, and the remedy is the epsilon, not the learning rate.

- **Off-policy contingency, so a future change re-decides instead of inheriting:** if
  ``num_iterations > 1``, async generation, or generation/optimizer misalignment ever lands, the
  clipping family wakes up. Re-decide then: DAPO's ``epsilon_high=0.28`` clip-higher, or
  ``loss_type="cispo"`` with ``epsilon_high=5.0`` (the larger-evidence ScaleRL recipe) -- and the
  disabled vLLM importance-sampling correction becomes a first-order question rather than a
  logged caveat.
"""

GRPO_LOSS_TYPE = "dr_grpo"
GRPO_SCALE_REWARDS = "none"
# Inert while runs stay on-policy (ratio == 1); see the off-policy contingency note above.
GRPO_EPSILON = 0.2

# TRL 1.10's accepted values, pinned so a typo fails at launch, not on a loaded rented GPU.
GRPO_LOSS_TYPES = ("grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo", "vespo")
GRPO_SCALE_REWARDS_MODES = ("none", "batch", "group")

# How TRL turns the vLLM-versus-trainer log-probability difference into a weight, when
# ``vllm_importance_sampling_correction`` is on. Two orthogonal axes (grpo_config.py:927-939):
# granularity, one ratio per token against one per sequence broadcast over that sequence's tokens;
# and constraint, clamping the ratio into [C_min, C_max] against zeroing it outside them. TRL's own
# default is the sequence-level masked one, so a whole rollout is discarded whenever its accumulated
# ratio leaves the band. Pinned here for the same reason as the loss types: TRL only refuses an
# unknown name inside the ratio arithmetic of the first generation batch, a card and a model load in.
VLLM_IMPORTANCE_SAMPLING_MODE = "sequence_mask"
VLLM_IMPORTANCE_SAMPLING_MODES = (
    "token_truncate",
    "token_mask",
    "sequence_truncate",
    "sequence_mask",
)
# The modes TRL requires beside ``loss_type="vespo"`` (grpo_trainer.py:924-928), which computes its
# own sequence-level weight and refuses a second sequence-level correction underneath it.
VESPO_IMPORTANCE_SAMPLING_MODES = ("token_truncate", "token_mask")

# HARNESS-SCAN-EXEMPT-multiline-comment-block -- these citations are the evidence the guard below
# rests on; a reader deciding whether to trust the tuple needs them here, next to it.
# Loss types whose EXECUTED aggregation under ``use_liger_kernel=True`` differs from their non-Liger
# (paper) semantics. Verified against the installed source (TRL 1.10.0, liger_kernel 0.8.1), branch
# by branch, on 2026-08-20:
#
# - ``dapo``, ``cispo``, ``vespo`` -- all three share Liger's DAPO normalizer branch
#   (liger_kernel/chunked_loss/grpo_loss.py:237-241). TRL's Liger call site never forwards
#   ``inputs["num_items_in_batch"]`` (trl/trainer/grpo_trainer.py:2955-2966, though the wrapper
#   accepts it, grpo_loss.py:502), so ``_compute_dapo_normalizer`` falls back to the CURRENT
#   MICRO-BATCH's token count (fused_linear_ppo.py:464) where the non-Liger branch divides by the
#   whole generation batch's (grpo_trainer.py:3209-3214). TRL then averages micro-batch losses
#   (grpo_trainer.py:2976-2977); at micro-batch 1 the composition is per-sequence original-GRPO
#   normalization -- the length bias DAPO and Dr. GRPO exist to remove.
# - ``luspo`` -- Liger's branch never multiplies by the loss mask and divides by batch x PADDED
#   length (grpo_loss.py:242-245: ``weighted = per_token_loss * attention_mask.sum(1, ...)``,
#   ``/ (B * weighted.shape[1])``), where TRL masks: ``(per_token_loss * mask).sum(-1).mean()``
#   (grpo_trainer.py:3220). On-policy, every padded position carries loss ``-advantage`` with a live
#   gradient through exp(logp - logp.detach()), so pads contribute value AND gradient. The two
#   coincide only at ``importance_sampling_level="sequence"`` (per_token_loss is (B, 1)); TRL only
#   WARNS about luspo at the "token" default (grpo_trainer.py:904-908), and no harness here sets the
#   level, so token-level is what would execute.
#
# The other four are faithful -- each Liger branch matches its non-Liger branch exactly:
# ``grpo``/``sapo`` (grpo_loss.py:224-228 vs grpo_trainer.py:3194-3198), ``bnpo`` (:229-231 vs
# :3199-3203), ``dr_grpo`` (:232-236 vs :3204-3208, ``max_completion_length`` forwarded at
# construction, grpo_trainer.py:1040-1047).
LIGER_UNFAITHFUL_LOSS_TYPES = ("dapo", "cispo", "vespo", "luspo")

# The subset a Liger run may name and get as described; what the refusal message offers.
LIGER_FAITHFUL_LOSS_TYPES = tuple(
    t for t in GRPO_LOSS_TYPES if t not in LIGER_UNFAITHFUL_LOSS_TYPES
)


def assert_known_estimator(loss_type: str, scale_rewards: str) -> None:
    """Refuse an estimator value TRL 1.10 does not know, at launch time.

    TRL itself only raises on an unknown `loss_type` at the first loss computation -- after the
    model is loaded, on what is usually a rented GPU.
    """
    if loss_type not in GRPO_LOSS_TYPES:
        raise ValueError(f"unknown {loss_type=}; TRL 1.10 accepts {GRPO_LOSS_TYPES}")
    if scale_rewards not in GRPO_SCALE_REWARDS_MODES:
        raise ValueError(f"unknown {scale_rewards=}; TRL 1.10 accepts {GRPO_SCALE_REWARDS_MODES}")


def assert_liger_faithful_estimator(
    loss_type: str, *, use_liger_kernel: bool, acknowledged: bool = False
) -> None:
    """Refuse a Liger-unfaithful ``loss_type`` at config construction, unless explicitly acknowledged.

    Every arm trained before 2026-08-20 recorded ``loss_type="dapo"`` while executing per-sequence
    original-GRPO aggregation (the mechanism above) and nothing said so. This is the guard that makes
    that combination impossible to reach silently: it either raises before any weights load, or the
    caller passed the explicit acknowledgement, which the caller records in its run config.

    An acknowledgement where nothing needs acknowledging also raises: a no-op flag left on a copied
    command line is a stale mental model of what the run executes, which is the same bug.
    """
    mismatched = use_liger_kernel and loss_type in LIGER_UNFAITHFUL_LOSS_TYPES
    if acknowledged and not mismatched:
        raise ValueError(
            f"acknowledge_liger_estimator_mismatch=True, but {loss_type=} with "
            f"{use_liger_kernel=} executes faithfully; there is no mismatch to acknowledge. "
            f"Drop the flag so its presence keeps meaning something."
        )
    if not mismatched or acknowledged:
        return
    mechanism = (
        "Liger never applies the loss mask at importance_sampling_level='token' (the default), so "
        "padded positions contribute value and gradient and the divisor is batch x padded length "
        "(liger grpo_loss.py:242-245 vs trl grpo_trainer.py:3220)"
        if loss_type == "luspo"
        else "TRL's Liger call site drops num_items_in_batch (trl grpo_trainer.py:2955-2966), so the "
        "normalizer falls back to the current micro-batch's token count (liger "
        "fused_linear_ppo.py:464) -- at micro-batch 1 that is per-sequence original-GRPO "
        "aggregation, not what the loss_type names"
    )
    raise ValueError(
        f"{loss_type=} with use_liger_kernel=True does not execute its paper semantics: {mechanism}. "
        f"Either switch to a Liger-faithful loss_type ({LIGER_FAITHFUL_LOSS_TYPES}), or state that "
        f"the executed estimator is the intended one by passing "
        f"acknowledge_liger_estimator_mismatch=True (--acknowledge-liger-estimator-mismatch), which "
        f"is recorded in the run config. See LIGER_UNFAITHFUL_LOSS_TYPES for the verified source "
        f"lines."
    )


def executed_estimator(
    loss_type: str, *, use_liger_kernel: bool, per_device_train_batch_size: int
) -> str:
    """Name the token-loss aggregation a run EXECUTES, as opposed to the one its config names.

    Pure provenance: the string lands in run records so a reader of an artifact never has to
    re-derive the Liger mechanism above to know what trained. The already-trained pre-2026-08-20
    arms are exactly the gap this closes -- their records say "dapo" and nothing says they executed
    per-sequence GRPO.
    """
    if loss_type not in GRPO_LOSS_TYPES:
        raise ValueError(f"unknown {loss_type=}; TRL 1.10 accepts {GRPO_LOSS_TYPES}")
    if per_device_train_batch_size < 1:
        raise ValueError(f"{per_device_train_batch_size=} cannot have produced a training batch")
    if not use_liger_kernel:
        return f"{loss_type} (non-Liger TRL path)"
    if loss_type not in LIGER_UNFAITHFUL_LOSS_TYPES:
        return f"{loss_type} (faithful under Liger)"
    if loss_type == "luspo":
        return "luspo -> unmasked pad-inclusive aggregation via Liger (loss mask not applied at token level)"
    if per_device_train_batch_size == 1:
        return f"{loss_type} -> per-sequence GRPO via Liger fallback (num_items_in_batch not forwarded)"
    return (
        f"{loss_type} -> per-micro-batch token normalization via Liger fallback "
        f"(num_items_in_batch not forwarded)"
    )
