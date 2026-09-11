"""The Liger-faithfulness guard and the executed-estimator provenance, `grpo/estimator_defaults.py`.

Both exist because every arm trained before 2026-08-20 recorded `loss_type="dapo"` while executing
per-sequence original-GRPO aggregation: TRL 1.10's Liger call site drops `num_items_in_batch`, and
nothing in the config surface or the artifacts said so. The guard makes the combination impossible
to reach silently; the provenance string makes what DID execute part of every run record.
"""

from __future__ import annotations

import pytest

from grpo.estimator_defaults import (
    GRPO_LOSS_TYPE,
    GRPO_LOSS_TYPES,
    GRPO_SCALE_REWARDS_MODES,
    LIGER_FAITHFUL_LOSS_TYPES,
    LIGER_UNFAITHFUL_LOSS_TYPES,
    assert_known_estimator,
    assert_liger_faithful_estimator,
    executed_estimator,
)


class TestTheLossTypePartition:
    def test_every_unfaithful_loss_type_is_a_real_trl_loss_type(self):
        assert set(LIGER_UNFAITHFUL_LOSS_TYPES) <= set(GRPO_LOSS_TYPES)

    def test_faithful_and_unfaithful_partition_the_menu(self):
        assert sorted(LIGER_FAITHFUL_LOSS_TYPES + LIGER_UNFAITHFUL_LOSS_TYPES) == sorted(
            GRPO_LOSS_TYPES
        )

    def test_the_repo_default_is_liger_faithful(self):
        """The whole point of the dr_grpo default: no acknowledgement needed on the standard path."""
        assert GRPO_LOSS_TYPE in LIGER_FAITHFUL_LOSS_TYPES


class TestAssertKnownEstimator:
    """The launch-time refusal of values TRL 1.10 would only reject after the model load.

    Promoted here from a private copy in ``games/train.py`` once ``reward_hacking/train.py`` grew a
    second, already-drifted copy; both trainers now call this one.
    """

    def test_an_unknown_loss_type_is_refused(self):
        with pytest.raises(ValueError, match="dpao"):
            assert_known_estimator("dpao", "none")

    def test_an_unknown_scale_rewards_mode_is_refused(self):
        with pytest.raises(ValueError, match="scale_rewards"):
            assert_known_estimator("dr_grpo", "batch-normalised")

    @pytest.mark.parametrize("loss_type", GRPO_LOSS_TYPES)
    @pytest.mark.parametrize("scale_rewards", GRPO_SCALE_REWARDS_MODES)
    def test_every_known_combination_passes(self, loss_type: str, scale_rewards: str):
        assert_known_estimator(loss_type, scale_rewards)


class TestAssertLigerFaithfulEstimator:
    @pytest.mark.parametrize("loss_type", ["dapo", "cispo", "vespo"])
    def test_the_num_items_family_is_refused_under_liger(self, loss_type: str):
        with pytest.raises(ValueError, match="num_items_in_batch"):
            assert_liger_faithful_estimator(loss_type, use_liger_kernel=True)

    def test_luspo_is_refused_under_liger_naming_its_own_mechanism(self):
        """luspo's defect is different -- the loss mask is never applied -- and the message says so."""
        with pytest.raises(ValueError, match="loss mask"):
            assert_liger_faithful_estimator("luspo", use_liger_kernel=True)

    def test_the_refusal_names_both_ways_out(self):
        with pytest.raises(ValueError, match="acknowledge_liger_estimator_mismatch") as excinfo:
            assert_liger_faithful_estimator("dapo", use_liger_kernel=True)
        assert "dr_grpo" in str(excinfo.value), "the message must offer a faithful loss_type"

    @pytest.mark.parametrize("loss_type", LIGER_FAITHFUL_LOSS_TYPES)
    def test_faithful_loss_types_pass_under_liger(self, loss_type: str):
        assert_liger_faithful_estimator(loss_type, use_liger_kernel=True)

    @pytest.mark.parametrize("loss_type", GRPO_LOSS_TYPES)
    def test_everything_passes_without_liger(self, loss_type: str):
        assert_liger_faithful_estimator(loss_type, use_liger_kernel=False)

    def test_the_acknowledgement_unblocks_an_unfaithful_combination(self):
        assert_liger_faithful_estimator("dapo", use_liger_kernel=True, acknowledged=True)

    def test_an_acknowledgement_with_nothing_to_acknowledge_is_itself_refused(self):
        """A no-op flag left on a copied command line is a stale mental model, so it fails loudly."""
        with pytest.raises(ValueError, match="no mismatch"):
            assert_liger_faithful_estimator("dr_grpo", use_liger_kernel=True, acknowledged=True)
        with pytest.raises(ValueError, match="no mismatch"):
            assert_liger_faithful_estimator("dapo", use_liger_kernel=False, acknowledged=True)


class TestExecutedEstimator:
    def test_faithful_under_liger_says_so(self):
        assert (
            executed_estimator("dr_grpo", use_liger_kernel=True, per_device_train_batch_size=1)
            == "dr_grpo (faithful under Liger)"
        )

    def test_non_liger_names_the_plain_trl_path(self):
        assert (
            executed_estimator("dapo", use_liger_kernel=False, per_device_train_batch_size=1)
            == "dapo (non-Liger TRL path)"
        )

    @pytest.mark.parametrize("loss_type", ["dapo", "cispo", "vespo"])
    def test_micro_batch_one_is_per_sequence_grpo(self, loss_type: str):
        """The production shape of every pre-2026-08-20 arm: micro-batch 1, Liger on."""
        assert executed_estimator(
            loss_type, use_liger_kernel=True, per_device_train_batch_size=1
        ) == (
            f"{loss_type} -> per-sequence GRPO via Liger fallback (num_items_in_batch not forwarded)"
        )

    @pytest.mark.parametrize("loss_type", ["dapo", "cispo", "vespo"])
    def test_larger_micro_batches_normalize_per_micro_batch(self, loss_type: str):
        assert executed_estimator(
            loss_type, use_liger_kernel=True, per_device_train_batch_size=8
        ) == (
            f"{loss_type} -> per-micro-batch token normalization via Liger fallback "
            f"(num_items_in_batch not forwarded)"
        )

    def test_luspo_under_liger_is_the_unmasked_aggregation_at_any_micro_batch(self):
        expected = (
            "luspo -> unmasked pad-inclusive aggregation via Liger "
            "(loss mask not applied at token level)"
        )
        assert (
            executed_estimator("luspo", use_liger_kernel=True, per_device_train_batch_size=1)
            == expected
        )
        assert (
            executed_estimator("luspo", use_liger_kernel=True, per_device_train_batch_size=8)
            == expected
        )

    def test_an_unknown_loss_type_is_refused_rather_than_described(self):
        with pytest.raises(ValueError, match="loss_type"):
            executed_estimator("dpao", use_liger_kernel=True, per_device_train_batch_size=1)

    def test_an_impossible_batch_size_is_refused(self):
        with pytest.raises(ValueError, match="per_device_train_batch_size"):
            executed_estimator("dr_grpo", use_liger_kernel=True, per_device_train_batch_size=0)
