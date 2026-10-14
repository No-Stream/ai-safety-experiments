"""Keep every GRPO trainer on the shared checkpoint retention policy."""

import ast
from pathlib import Path

import pytest

from reward_hacking.semantic_identity import semantic_config_value


def test_retention_settings_do_not_change_training_semantics() -> None:
    config = {"learning_rate": 1e-5, "checkpoint_keep_latest": 2, "checkpoint_keep_middle": 3}
    assert semantic_config_value(config) == {"learning_rate": 1e-5}


@pytest.mark.parametrize(
    ("module_path", "trainer_class"),
    [
        ("games/train.py", "InstrumentedGRPOTrainer"),
        ("grpo/rlvr_math.py", "CheckpointRetainingGRPOTrainer"),
        ("grpo/throughput.py", "CheckpointRetainingGRPOTrainer"),
    ],
)
def test_grpo_trainers_inherit_retention_before_trl(module_path: str, trainer_class: str) -> None:
    tree = ast.parse(Path(module_path).read_text(encoding="utf-8"))
    trainers = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == trainer_class
    ]
    assert len(trainers) == 1
    assert [ast.unparse(base) for base in trainers[0].bases] == [
        "CheckpointRetentionMixin",
        "GRPOTrainer",
    ]


@pytest.mark.parametrize(
    "module_path", ["games/train.py", "reward_hacking/train.py", "grpo/rlvr_math.py"]
)
def test_training_builders_forward_retention_settings(module_path: str) -> None:
    tree = ast.parse(Path(module_path).read_text(encoding="utf-8"))
    assignments = {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
    }
    config_name = "cfg" if module_path == "grpo/rlvr_math.py" else "config"
    for field in ("checkpoint_keep_latest", "checkpoint_keep_middle"):
        assert assignments[f"trainer.{field}"] == f"{config_name}.{field}"
    rotation_settings = [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) in {"GRPOConfig", "config_class"}
        for keyword in node.keywords
        if keyword.arg == "save_total_limit"
    ]
    assert rotation_settings
    assert all(ast.literal_eval(value) is None for value in rotation_settings)
