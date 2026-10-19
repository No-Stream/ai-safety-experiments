"""Keep every GRPO trainer on the shared checkpoint retention policy."""

import pytest
from trl import GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]

from games.train import InstrumentedGRPOTrainer
from grpo.rlvr_math import CheckpointRetainingGRPOTrainer as RLVRCheckpointRetainingGRPOTrainer
from grpo.throughput import (
    CheckpointRetainingGRPOTrainer as ThroughputCheckpointRetainingGRPOTrainer,
)
from reward_hacking.checkpoint_retention import CheckpointRetentionMixin
from reward_hacking.semantic_identity import semantic_config_value


def test_retention_settings_do_not_change_training_semantics() -> None:
    config = {"learning_rate": 1e-5, "checkpoint_keep_latest": 2, "checkpoint_keep_middle": 3}
    assert semantic_config_value(config) == {"learning_rate": 1e-5}


@pytest.mark.parametrize(
    "trainer_class",
    [
        InstrumentedGRPOTrainer,
        RLVRCheckpointRetainingGRPOTrainer,
        ThroughputCheckpointRetainingGRPOTrainer,
    ],
)
def test_grpo_trainers_use_retention_mixin_before_transformers(
    trainer_class: type[GRPOTrainer],
) -> None:
    method_resolution_order = trainer_class.__mro__
    assert issubclass(trainer_class, CheckpointRetentionMixin)
    assert method_resolution_order.index(CheckpointRetentionMixin) < method_resolution_order.index(
        GRPOTrainer
    )
