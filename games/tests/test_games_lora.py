"""Pin the four things about checkpoint handling that fail silently rather than loudly.

Offline and CPU-only. Nothing here downloads anything or loads real weights; the two classes that
need a model at all build one out of a handful of 8x8 tensors, which is enough because every trap
below is about *names*, not numbers. Each guard catches a mistake whose symptom is a
plausible-looking result rather than a crash, which is exactly why each needs a test that has been
watched to fail.

:class:`TestCheckpointOrderIsNumeric` is the sort trap. With `save_steps=10` a run reaches
`checkpoint-100`, which sorts before `checkpoint-20` as text, so a lexical sort would hand every
before/after comparison its checkpoints in the wrong order and invert the trend.

:class:`TestAdapterBaseMismatch` is the wrong-base trap. Sibling models in one family share layer
names and shapes, so an adapter trained on the 4B loads onto the 2B without complaint and produces
a model that generates fluent text and is not the one you trained.

:class:`TestAdapterMustActuallyApply` is the wrong-tree trap, and it is the one that actually bit.
PEFT matches adapter weights to modules by name, so a base built with a different module tree than
training used matches nothing -- and PEFT only warns. The merge then folds in nothing and writes
out the base model under a trained checkpoint's name. `test_the_merge_is_silent_about_it` is the
sabotage: it shows the failure happening quietly, so the guard has something to be load-bearing
against.

:class:`TestExportSelfConsistency` is the unloadable-export trap. transformers renames keys as it
writes, so an export can end up naming a text-only class in its config while its weights keep a
composite checkpoint's nesting. Nothing can load that, and the error a consumer raises names a
missing module rather than the mismatch.

:class:`TestRuntimeAdapterSeam` covers the other seam, the one an activation capture uses instead of
merging. Its load-bearing claim is that PEFT adapts the model tree *in place*, so a forward pass
driven through the model object the caller built applies the adapter -- which is what lets a
checkpoint ladder keep one base in memory. `test_forwarding_the_inner_model_applies_the_adapter` is
that claim as a test rather than as an argument, and `test_re_pointing_keeps_one_base` is the
re-point path a ladder actually takes. :class:`TestOneAdapterConfig` covers why re-pointing needs a
guard: PEFT reads an adapter's config only when the adapter *name* is new, so a later checkpoint with
a different rank is silently loaded under the first one's config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import save_file
from torch import nn
from transformers import PretrainedConfig
from trl.trainer import grpo_trainer

from games import lora
from games.lora import (
    ADAPTER_CONFIG_FILENAME,
    CONFIG_FILENAME,
    MERGE_ADAPTER_NAME,
    PROCESSOR_CONFIG_FILENAME,
    TEXT_SUBMODEL_KEY,
    assert_adapter_applies,
    assert_adapter_matches_base,
    assert_export_is_self_consistent,
    checkpoint_step,
    export_merged_checkpoint,
    iter_checkpoints,
    read_adapter_base_model,
    read_export_weight_names,
)

BASE_MODEL = "Qwen/Qwen3.5-4B"
SIBLING_MODEL = "Qwen/Qwen3.5-2B"

WIDTH = 8
ADAPTED_MODULES = ("q_proj", "o_proj")
TINY_BASE_MODEL = "tiny/composite-base"


def make_adapter_dir(root: Path, base_model_name_or_path: str | None) -> Path:
    """Lay down a minimal PEFT adapter directory: only the config the guard reads."""
    adapter_dir = root / "checkpoint-10"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {"peft_type": "LORA", "r": 16}
    if base_model_name_or_path is not None:
        config["base_model_name_or_path"] = base_model_name_or_path
    (adapter_dir / ADAPTER_CONFIG_FILENAME).write_text(json.dumps(config))
    return adapter_dir


def set_adapter_config_field(adapter_dir: Path, field: str, value: object) -> None:
    """Rewrite one field of an already-written adapter config."""
    config_path = adapter_dir / ADAPTER_CONFIG_FILENAME
    config = cast("dict[str, object]", json.loads(config_path.read_text()))
    config[field] = value
    config_path.write_text(json.dumps(config))


class TinyAttention(nn.Module):
    """Two projections named the way the real models name them, since names are the whole trap."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(WIDTH, WIDTH, bias=False)
        self.o_proj = nn.Linear(WIDTH, WIDTH, bias=False)


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = TinyAttention()


class TinyTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer()])


class TinyTextOnlyModel(nn.Module):
    """The flat tree: `model.layers.0.self_attn.q_proj`, as a text-only auto class builds it."""

    def __init__(self) -> None:
        super().__init__()
        self.model = TinyTower()
        self.config = PretrainedConfig()


class TinyTextTowerHolder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = TinyTower()


class TinyCompositeModel(nn.Module):
    """The nested tree: `model.language_model.layers.0.self_attn.q_proj`, as training builds it.

    Carries a `text_config` for the same reason the real one does: it is what marks a checkpoint
    composite, and so what decides whether a processor has to be exported beside the weights.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = TinyTextTowerHolder()
        config = PretrainedConfig()
        config.text_config = PretrainedConfig()
        self.config = config


COMPOSITE_LAYER_PATH = f"model.{TEXT_SUBMODEL_KEY}.layers.0"
TEXT_ONLY_LAYER_PATH = "model.layers.0"


def tiny_lora_config() -> LoraConfig:
    return LoraConfig(r=2, target_modules=list(ADAPTED_MODULES))


def train_tiny_adapter(root: Path) -> Path:
    """Save an adapter off the nested tree, with non-zero B so that merging it must change weights.

    LoRA initialises B to zeros, so an adapter straight out of `get_peft_model` merges to a no-op
    whether or not its names matched. Nudging B is what makes "the weights changed" a real signal.
    """
    adapter_dir = root / "checkpoint-70"
    trained = get_peft_model(cast("Any", TinyCompositeModel()), tiny_lora_config())
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if "lora_B" in name:
                parameter.add_(0.5)
    trained.save_pretrained(str(adapter_dir))
    config_path = adapter_dir / ADAPTER_CONFIG_FILENAME
    config = cast("dict[str, object]", json.loads(config_path.read_text()))
    config["base_model_name_or_path"] = TINY_BASE_MODEL
    config_path.write_text(json.dumps(config))
    return adapter_dir


class StubTokenizer:
    """Stands in for the tokenizer the merge fetches, which no test here has an opinion about."""

    @staticmethod
    def from_pretrained(model_id: str, **kwargs: object) -> str:
        del kwargs
        return f"tokenizer-for-{model_id}"


class StubProcessorLoader:
    """Stands in for the processor a composite base is fetched with."""

    @staticmethod
    def from_pretrained(model_id: str, **kwargs: object) -> str:
        del kwargs
        return f"processor-for-{model_id}"


def adapted_weight(model: nn.Module, layer_path: str) -> torch.Tensor:
    """Return the q_proj weight at `layer_path`, which is where the merge either lands or does not.

    Addressed by dotted path rather than by attribute walking, because the path *is* the thing
    under test: the two trees differ only in whether `language_model` sits in the middle of it.
    """
    layer = model.get_submodule(layer_path)
    assert isinstance(layer, TinyLayer)
    return layer.self_attn.q_proj.weight


def write_export(
    root: Path,
    *,
    composite_config: bool,
    nested_weights: bool,
    sharded: bool = False,
    with_processor: bool = True,
) -> Path:
    """Write the files a consumer reads to decide how to load a directory, and nothing else."""
    root.mkdir(parents=True, exist_ok=True)
    architecture = "Qwen3_5ForConditionalGeneration" if composite_config else "Qwen3_5ForCausalLM"
    config: dict[str, object] = {"architectures": [architecture], "model_type": "qwen3_5"}
    if composite_config:
        config["text_config"] = {"model_type": "qwen3_5_text"}
        if with_processor:
            (root / PROCESSOR_CONFIG_FILENAME).write_text(
                json.dumps({"processor_class": "Qwen3VLProcessor"})
            )
    (root / CONFIG_FILENAME).write_text(json.dumps(config))
    prefix = f"model.{TEXT_SUBMODEL_KEY}" if nested_weights else "model"
    tensors: dict[str, torch.Tensor] = {
        f"{prefix}.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2)
    }
    if sharded:
        shard_name = "model-00001-of-00001.safetensors"
        save_file(tensors, str(root / shard_name))
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": dict.fromkeys(tensors, shard_name)})
        )
    else:
        save_file(tensors, str(root / "model.safetensors"))
    return root


class TestCheckpointOrderIsNumeric:
    def test_steps_come_back_in_training_order_not_lexical_order(self, tmp_path: Path) -> None:
        for step in (100, 20, 5, 10):
            (tmp_path / f"checkpoint-{step}").mkdir()
        assert [path.name for path in iter_checkpoints(tmp_path)] == [
            "checkpoint-5",
            "checkpoint-10",
            "checkpoint-20",
            "checkpoint-100",
        ]

    def test_lexical_sorting_would_have_disagreed(self, tmp_path: Path) -> None:
        """Guards the guard: if these ever agreed, the test above would prove nothing."""
        for step in (100, 20, 5, 10):
            (tmp_path / f"checkpoint-{step}").mkdir()
        numeric = [path.name for path in iter_checkpoints(tmp_path)]
        lexical = sorted(numeric)
        assert numeric != lexical

    def test_non_checkpoint_entries_are_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoint-10").mkdir()
        (tmp_path / "completions").mkdir()
        (tmp_path / "runs").mkdir()
        (tmp_path / "train_summary.json").write_text("{}")
        (tmp_path / "checkpoint-notanumber").mkdir()
        assert [path.name for path in iter_checkpoints(tmp_path)] == ["checkpoint-10"]

    def test_a_run_with_no_checkpoints_yields_nothing(self, tmp_path: Path) -> None:
        assert list(iter_checkpoints(tmp_path)) == []

    def test_a_missing_run_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="is not a directory"):
            list(iter_checkpoints(tmp_path / "nope"))

    def test_checkpoint_step_parses_and_rejects(self, tmp_path: Path) -> None:
        assert checkpoint_step(tmp_path / "checkpoint-40") == 40
        with pytest.raises(ValueError, match="not a checkpoint directory"):
            checkpoint_step(tmp_path / "final-model")


class TestAdapterBaseMismatch:
    def test_a_matching_base_passes(self, tmp_path: Path) -> None:
        adapter_dir = make_adapter_dir(tmp_path, BASE_MODEL)
        assert_adapter_matches_base(adapter_dir, BASE_MODEL)

    def test_a_trailing_slash_is_not_a_mismatch(self, tmp_path: Path) -> None:
        adapter_dir = make_adapter_dir(tmp_path, f"{BASE_MODEL}/")
        assert_adapter_matches_base(adapter_dir, BASE_MODEL)

    def test_a_sibling_model_in_the_same_family_raises(self, tmp_path: Path) -> None:
        """The case that would otherwise load happily: same architecture, different size."""
        adapter_dir = make_adapter_dir(tmp_path, BASE_MODEL)
        with pytest.raises(ValueError, match="was trained against"):
            assert_adapter_matches_base(adapter_dir, SIBLING_MODEL)

    def test_the_message_names_both_models(self, tmp_path: Path) -> None:
        adapter_dir = make_adapter_dir(tmp_path, BASE_MODEL)
        with pytest.raises(ValueError, match="was trained against") as caught:
            assert_adapter_matches_base(adapter_dir, SIBLING_MODEL)
        assert BASE_MODEL in str(caught.value)
        assert SIBLING_MODEL in str(caught.value)

    def test_a_directory_without_an_adapter_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not a PEFT adapter directory"):
            read_adapter_base_model(tmp_path)

    @pytest.mark.parametrize("recorded", [None, ""])
    def test_a_config_with_no_recorded_base_raises(
        self, tmp_path: Path, recorded: str | None
    ) -> None:
        adapter_dir = make_adapter_dir(tmp_path, recorded)
        with pytest.raises(ValueError, match="no base_model_name_or_path"):
            read_adapter_base_model(adapter_dir)


class TestMergeLoadsTheBaseTheWayTrainingDoes:
    def test_the_merge_and_the_trainer_share_one_loader(self) -> None:
        """The invariant behind the whole bug, pinned where it costs no weights to check.

        Adapter names are only meaningful against the module tree they were trained on, so the
        merge cannot pick its own way to build the base. Sharing `GRPOTrainer`'s loader is what
        makes the two trees identical by construction; an auto class chosen here independently
        resolves to a different class for this model family and every adapter name then misses.
        """
        assert (
            lora.create_model_from_path is grpo_trainer.create_model_from_path  # pyright: ignore[reportPrivateImportUsage]
        )

    def test_the_merge_actually_calls_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The companion the identity check needs: that name has to be the one the merge calls.

        Pinning only the import leaves a reverted call site invisible, which is how a guard ends
        up watching a branch nothing reaches. Standing a loader in here exercises the call itself,
        and with it the whole of `load_merged_model` on a tree small enough to hold in mind.
        """
        adapter_dir = train_tiny_adapter(tmp_path)
        requested: list[str] = []

        def stub_loader(model_id: str, **kwargs: object) -> nn.Module:
            del kwargs
            requested.append(model_id)
            return TinyCompositeModel()

        monkeypatch.setattr(lora, "create_model_from_path", stub_loader)
        monkeypatch.setattr(lora, "AutoTokenizer", StubTokenizer)
        monkeypatch.setattr(lora, "AutoProcessor", StubProcessorLoader)
        merged = lora.load_merged_model(adapter_dir, TINY_BASE_MODEL)
        assert requested == [TINY_BASE_MODEL]
        assert merged.applied_adapter_weights == 2 * len(ADAPTED_MODULES)
        assert merged.processor == f"processor-for-{TINY_BASE_MODEL}"


class TestAdapterMustActuallyApply:
    """The trap that bit: an adapter whose names do not exist in the tree the merge built."""

    def test_a_matching_tree_applies_every_weight(self, tmp_path: Path) -> None:
        adapter_dir = train_tiny_adapter(tmp_path)
        adapted = PeftModel.from_pretrained(
            TinyCompositeModel(), str(adapter_dir), adapter_name=MERGE_ADAPTER_NAME
        )
        applied = assert_adapter_applies(adapted, adapter_dir)
        assert applied == 2 * len(ADAPTED_MODULES)

    def test_a_matching_tree_really_changes_the_weights(self, tmp_path: Path) -> None:
        """The check that cannot be satisfied by a merge that ran and did nothing."""
        adapter_dir = train_tiny_adapter(tmp_path)
        base = TinyCompositeModel()
        before = adapted_weight(base, COMPOSITE_LAYER_PATH).detach().clone()
        adapted = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name=MERGE_ADAPTER_NAME)
        assert_adapter_applies(adapted, adapter_dir)
        merged = adapted.merge_and_unload()  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        assert not torch.equal(before, adapted_weight(merged, COMPOSITE_LAYER_PATH))

    def test_a_tree_without_the_nesting_raises(self, tmp_path: Path) -> None:
        """The reproduction: same leaf names, one level of nesting missing, so nothing matches."""
        adapter_dir = train_tiny_adapter(tmp_path)
        with pytest.warns(UserWarning, match="missing adapter keys"):
            adapted = PeftModel.from_pretrained(
                TinyTextOnlyModel(), str(adapter_dir), adapter_name=MERGE_ADAPTER_NAME
            )
        with pytest.raises(ValueError, match="did not apply") as caught:
            assert_adapter_applies(adapted, adapter_dir)
        message = str(caught.value)
        assert "0 of 4 adapter weights landed" in message
        assert TinyTextOnlyModel.__name__ in message
        assert TEXT_SUBMODEL_KEY in message

    def test_the_merge_is_silent_about_it(self, tmp_path: Path) -> None:
        """Guards the guard: without it, the merge folds in nothing and says so only as a warning.

        If this ever fails, PEFT started refusing the mismatch itself and the guard above is no
        longer what stands between us and a base model wearing a checkpoint's name.
        """
        adapter_dir = train_tiny_adapter(tmp_path)
        base = TinyTextOnlyModel()
        before = adapted_weight(base, TEXT_ONLY_LAYER_PATH).detach().clone()
        with pytest.warns(UserWarning, match="missing adapter keys"):
            adapted = PeftModel.from_pretrained(
                base, str(adapter_dir), adapter_name=MERGE_ADAPTER_NAME
            )
        merged = adapted.merge_and_unload()  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        assert torch.equal(before, adapted_weight(merged, TEXT_ONLY_LAYER_PATH))


class TestRuntimeAdapterSeam:
    """Applying an adapter at runtime, which is what a capture does instead of merging."""

    def test_attach_reports_the_weights_that_landed(self, tmp_path: Path) -> None:
        adapter_dir = train_tiny_adapter(tmp_path)
        attached = lora.attach_adapter(
            cast("Any", TinyCompositeModel()), adapter_dir, TINY_BASE_MODEL
        )
        assert attached.applied_adapter_weights == 2 * len(ADAPTED_MODULES)

    def test_forwarding_the_inner_model_applies_the_adapter(self, tmp_path: Path) -> None:
        """The claim the capture rests on: PEFT rewrites the tree it was handed, in place.

        If this ever fails, the capture driver is reading pristine base activations through the model
        object it built while the PeftModel wrapper holds the trained weights, and no guard anywhere
        would notice -- so this is the test that keeps that from being an assumption.
        """
        adapter_dir = train_tiny_adapter(tmp_path)
        base = TinyCompositeModel()
        projection = cast("Any", base).model.language_model.layers[0].self_attn.q_proj
        inputs = torch.ones(1, WIDTH)
        before = projection(inputs).detach().clone()
        lora.attach_adapter(cast("Any", base), adapter_dir, TINY_BASE_MODEL)
        after = cast("Any", base).model.language_model.layers[0].self_attn.q_proj(inputs)
        assert not torch.allclose(before, after)

    def test_re_pointing_keeps_one_base(self, tmp_path: Path) -> None:
        """A ladder re-points one adapter name per checkpoint rather than rebuilding the base."""
        first = train_tiny_adapter(tmp_path / "run-a")
        second = train_tiny_adapter(tmp_path / "run-b")
        base = TinyCompositeModel()
        attached = lora.attach_adapter(cast("Any", base), first, TINY_BASE_MODEL)
        again = lora.attach_adapter(
            cast("Any", base), second, TINY_BASE_MODEL, existing=attached.peft_model
        )
        assert again.peft_model is attached.peft_model
        assert again.applied_adapter_weights == 2 * len(ADAPTED_MODULES)

    def test_a_mismatched_base_is_refused_before_any_load(self, tmp_path: Path) -> None:
        adapter_dir = make_adapter_dir(tmp_path, BASE_MODEL)
        with pytest.raises(ValueError, match="was trained against"):
            lora.attach_adapter(cast("Any", TinyCompositeModel()), adapter_dir, SIBLING_MODEL)

    def test_a_tree_without_the_nesting_raises(self, tmp_path: Path) -> None:
        adapter_dir = train_tiny_adapter(tmp_path)
        with (
            pytest.warns(UserWarning, match="missing adapter keys"),
            pytest.raises(ValueError, match="did not apply"),
        ):
            lora.attach_adapter(cast("Any", TinyTextOnlyModel()), adapter_dir, TINY_BASE_MODEL)


class TestOneAdapterConfig:
    """A ladder that re-points one adapter name has to agree on what a checkpoint means."""

    def test_identical_configs_pass(self, tmp_path: Path) -> None:
        dirs = [make_adapter_dir(tmp_path / name, TINY_BASE_MODEL) for name in ("a", "b")]
        lora.assert_one_adapter_config(dirs)

    def test_a_differing_rank_is_refused(self, tmp_path: Path) -> None:
        first = make_adapter_dir(tmp_path / "a", TINY_BASE_MODEL)
        second = make_adapter_dir(tmp_path / "b", TINY_BASE_MODEL)
        set_adapter_config_field(second, "r", 32)
        with pytest.raises(ValueError, match=r"differ between .* on \['r'\]"):
            lora.assert_one_adapter_config([first, second])

    def test_a_differing_target_list_is_refused(self, tmp_path: Path) -> None:
        first = make_adapter_dir(tmp_path / "a", TINY_BASE_MODEL)
        second = make_adapter_dir(tmp_path / "b", TINY_BASE_MODEL)
        set_adapter_config_field(second, "target_modules", ["q_proj"])
        with pytest.raises(ValueError, match="target_modules"):
            lora.assert_one_adapter_config([first, second])

    def test_reordered_target_modules_are_one_config(self, tmp_path: Path) -> None:
        """PEFT holds `target_modules` as a set, so successive saves permute the JSON array.

        Three orderings of an identical config were observed across the 14 checkpoints of one
        ladder; a guard that compares the raw arrays refuses that uniform ladder outright.
        """
        first = make_adapter_dir(tmp_path / "a", TINY_BASE_MODEL)
        second = make_adapter_dir(tmp_path / "b", TINY_BASE_MODEL)
        set_adapter_config_field(first, "target_modules", ["q_proj", "o_proj"])
        set_adapter_config_field(second, "target_modules", ["o_proj", "q_proj"])
        lora.assert_one_adapter_config([first, second])

    def test_reordering_tolerance_does_not_mask_a_real_change(self, tmp_path: Path) -> None:
        first = make_adapter_dir(tmp_path / "a", TINY_BASE_MODEL)
        second = make_adapter_dir(tmp_path / "b", TINY_BASE_MODEL)
        set_adapter_config_field(first, "target_modules", ["q_proj", "o_proj"])
        set_adapter_config_field(second, "target_modules", ["o_proj", "k_proj"])
        with pytest.raises(ValueError, match="target_modules"):
            lora.assert_one_adapter_config([first, second])

    def test_the_identity_reads_the_fields_that_decide_meaning(self, tmp_path: Path) -> None:
        adapter_dir = make_adapter_dir(tmp_path, TINY_BASE_MODEL)
        identity = lora.adapter_config_identity(adapter_dir)
        assert set(identity) == set(lora.ADAPTER_CONFIG_IDENTITY_FIELDS)
        assert identity["r"] == 16
        assert identity["base_model_name_or_path"] == TINY_BASE_MODEL


class TestExportSelfConsistency:
    """Whether the written directory's config and its weight names describe the same model."""

    def test_a_composite_config_with_nested_weights_passes(self, tmp_path: Path) -> None:
        export = write_export(tmp_path / "merged", composite_config=True, nested_weights=True)
        assert_export_is_self_consistent(export)

    def test_a_text_only_config_with_flat_weights_passes(self, tmp_path: Path) -> None:
        export = write_export(tmp_path / "merged", composite_config=False, nested_weights=False)
        assert_export_is_self_consistent(export)

    def test_a_text_only_config_with_nested_weights_raises(self, tmp_path: Path) -> None:
        """The export the broken merge actually wrote, which vLLM refused for an unrelated-looking
        reason."""
        export = write_export(tmp_path / "merged", composite_config=False, nested_weights=True)
        with pytest.raises(ValueError, match="not internally consistent") as caught:
            assert_export_is_self_consistent(export)
        assert "Qwen3_5ForCausalLM" in str(caught.value)

    def test_a_composite_config_with_flat_weights_raises(self, tmp_path: Path) -> None:
        export = write_export(tmp_path / "merged", composite_config=True, nested_weights=False)
        with pytest.raises(ValueError, match="not internally consistent"):
            assert_export_is_self_consistent(export)

    def test_a_sharded_export_is_read_through_its_index(self, tmp_path: Path) -> None:
        """Bigger models shard, and the names then live in the index rather than in a lone file."""
        export = write_export(
            tmp_path / "merged", composite_config=True, nested_weights=True, sharded=True
        )
        assert read_export_weight_names(export) == [
            f"model.{TEXT_SUBMODEL_KEY}.layers.0.self_attn.q_proj.weight"
        ]
        assert_export_is_self_consistent(export)

    def test_a_directory_holding_no_weights_raises(self, tmp_path: Path) -> None:
        bare = tmp_path / "merged"
        bare.mkdir()
        (bare / CONFIG_FILENAME).write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"]}))
        with pytest.raises(FileNotFoundError, match="not a saved model directory"):
            assert_export_is_self_consistent(bare)

    def test_a_directory_holding_no_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="has no architecture"):
            assert_export_is_self_consistent(tmp_path)

    def test_a_composite_export_missing_its_processor_config_raises(self, tmp_path: Path) -> None:
        export = write_export(
            tmp_path / "merged",
            composite_config=True,
            nested_weights=True,
            with_processor=False,
        )
        with pytest.raises(ValueError, match=f"holds no {PROCESSOR_CONFIG_FILENAME}"):
            assert_export_is_self_consistent(export)


class FakeMergedModel:
    """Writes what a real model's save_pretrained writes: the config and the weights, no processor."""

    def __init__(self, *, composite_config: bool, nested_weights: bool) -> None:
        self.composite_config = composite_config
        self.nested_weights = nested_weights

    def save_pretrained(self, path: str) -> None:
        write_export(
            Path(path),
            composite_config=self.composite_config,
            nested_weights=self.nested_weights,
            with_processor=False,
        )


class FakeTokenizer:
    def save_pretrained(self, path: str) -> None:
        (Path(path) / "tokenizer_config.json").write_text(json.dumps({}))


class FakeProcessor:
    def save_pretrained(self, path: str) -> None:
        (Path(path) / PROCESSOR_CONFIG_FILENAME).write_text(
            json.dumps({"processor_class": "Qwen3VLProcessor"})
        )


class TestExportIsValidatedBeforeItLands:
    """Whether a directory at the consumer's path can ever be an unvalidated one."""

    @staticmethod
    def stub_merge(
        *, composite_config: bool, nested_weights: bool, with_processor: bool = True
    ) -> object:
        def load(adapter_dir: Path, base_model_id: str, **kwargs: object) -> lora.MergedModel:
            del adapter_dir, base_model_id, kwargs
            return lora.MergedModel(
                model=cast(
                    "Any",
                    FakeMergedModel(
                        composite_config=composite_config, nested_weights=nested_weights
                    ),
                ),
                tokenizer=cast("Any", FakeTokenizer()),
                applied_adapter_weights=4,
                processor=cast("Any", FakeProcessor()) if with_processor else None,
            )

        return load

    def test_a_consistent_export_is_moved_into_place(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            lora,
            "load_merged_model",
            self.stub_merge(composite_config=True, nested_weights=True),
        )
        out_dir = tmp_path / "merged"
        assert export_merged_checkpoint(tmp_path / "checkpoint-70", BASE_MODEL, out_dir) == out_dir
        assert (out_dir / CONFIG_FILENAME).is_file()
        assert (out_dir / "tokenizer_config.json").is_file()
        assert not (tmp_path / f"merged{lora.STAGING_SUFFIX}").exists()

    def test_an_inconsistent_export_never_reaches_the_output_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of staging: the directory a consumer would read is never written."""
        monkeypatch.setattr(
            lora,
            "load_merged_model",
            self.stub_merge(composite_config=False, nested_weights=True),
        )
        out_dir = tmp_path / "merged"
        with pytest.raises(ValueError, match="not internally consistent"):
            export_merged_checkpoint(tmp_path / "checkpoint-70", BASE_MODEL, out_dir)
        assert not out_dir.exists() or not any(out_dir.iterdir())
        assert (tmp_path / f"merged{lora.STAGING_SUFFIX}").is_dir()

    def test_a_composite_export_without_its_processor_never_lands(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second failure this path had: coherent weights, but nothing for vLLM's processor.

        A composite config sends a consumer looking for an image processor before it reads a weight,
        so a model-and-tokenizer export dies on a missing preprocessor and blames the processor
        rather than the merge. Caught here so it cannot be caught by a GPU box again.
        """
        monkeypatch.setattr(
            lora,
            "load_merged_model",
            self.stub_merge(composite_config=True, nested_weights=True, with_processor=False),
        )
        out_dir = tmp_path / "merged"
        with pytest.raises(ValueError, match=f"holds no {PROCESSOR_CONFIG_FILENAME}"):
            export_merged_checkpoint(tmp_path / "checkpoint-70", BASE_MODEL, out_dir)
        assert not out_dir.exists() or not any(out_dir.iterdir())

    def test_a_text_only_export_needs_no_processor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other side of it: a text-only config promises no processor, so none is required."""
        monkeypatch.setattr(
            lora,
            "load_merged_model",
            self.stub_merge(composite_config=False, nested_weights=False, with_processor=False),
        )
        out_dir = tmp_path / "merged"
        assert export_merged_checkpoint(tmp_path / "checkpoint-70", BASE_MODEL, out_dir) == out_dir

    def test_a_leftover_staging_directory_is_not_written_over(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            lora,
            "load_merged_model",
            self.stub_merge(composite_config=True, nested_weights=True),
        )
        staging = tmp_path / f"merged{lora.STAGING_SUFFIX}"
        staging.mkdir()
        with pytest.raises(FileExistsError, match="left over from an export"):
            export_merged_checkpoint(tmp_path / "checkpoint-70", BASE_MODEL, tmp_path / "merged")


class TestExportRefusesToOverwrite:
    def test_a_non_empty_output_directory_raises_before_any_load(self, tmp_path: Path) -> None:
        """Checked before the base model is fetched, so this needs no weights."""
        adapter_dir = make_adapter_dir(tmp_path, BASE_MODEL)
        out_dir = tmp_path / "merged"
        out_dir.mkdir()
        (out_dir / "config.json").write_text("{}")
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            export_merged_checkpoint(adapter_dir, BASE_MODEL, out_dir)

    def test_the_mismatch_guard_runs_before_any_load_too(self, tmp_path: Path) -> None:
        adapter_dir = make_adapter_dir(tmp_path, BASE_MODEL)
        with pytest.raises(ValueError, match="was trained against"):
            export_merged_checkpoint(adapter_dir, SIBLING_MODEL, tmp_path / "merged")
