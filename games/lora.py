"""Merge a LoRA adapter into its base model, so every downstream consumer sees a plain HF dir.

Training saves adapters; the eval battery, the interp stack, `HFBackend`, and `VLLMBackend` all
want an ordinary model directory. Merging once per checkpoint keeps that seam narrow: nothing
downstream needs to know an adapter was ever involved, and no consumer grows a PEFT code path.

Four things here fail silently rather than loudly, and each has a guard.

The adapter/base match: a LoRA adapter loads happily onto the wrong base of the same architecture
-- same layer names, same shapes -- and produces a model that runs, generates fluent text, and is
not the thing you trained. `assert_adapter_matches_base` reads the base model the adapter recorded
at training time and refuses to merge onto anything else.

The module tree: PEFT matches adapter weights to modules *by name*, so the base has to be built
the way training built it or no name lines up. That is not the same as "same model id". For a
Qwen3.5 checkpoint, `AutoModelForCausalLM` resolves the text-only `Qwen3_5ForCausalLM`, whose
layers sit at `model.layers.*`, while TRL builds the class the hub config names --
`Qwen3_5ForConditionalGeneration` -- whose text tower sits at `model.language_model.layers.*`.
Adapters trained through TRL therefore carry the nested names, none of which exist in the flat
tree. So the base is loaded here through the same `create_model_from_path` that `GRPOTrainer`
calls, which makes the two trees identical by construction rather than by agreement.

Whether the adapter applied at all: when no name lines up, PEFT emits a `UserWarning` and carries
on, `merge_and_unload` merges nothing, and the export is the untouched base model wearing a
trained checkpoint's name -- a null result nothing downstream can distinguish from a real one.
`assert_adapter_applies` turns that into an exception.

Whether the written directory is coherent: transformers renames keys on the way out, so what
`save_pretrained` writes is not always what the model held in memory. An export whose config names
a text-only class while its weights keep the nested names loads under neither reading, and a
composite config additionally commits the export to carrying a processor -- vLLM builds a
multimodal processor from it before reading a single weight, so a model-and-tokenizer directory
dies on a missing image processor. `assert_export_is_self_consistent` reads the files back and
refuses to hand on a directory that fails either way.

There is no text-only escape from that, incidentally: in transformers 5.15 the text-only class
writes the nested names too, so a text-only export of this family is not a thing the library can
produce. Composite is the only coherent shape, and it has to be paid for in full.

Checkpoint ordering is the last quiet trap: `sorted()` on `checkpoint-100` and `checkpoint-20`
puts 100 first, which would silently reverse a before/after trend. `iter_checkpoints` sorts on
the parsed step number.

Merging is not always the right seam, and `load_adapter_base` / `attach_adapter` are the other
one. A merge rounds `W + BA` back into bf16, and because a rank-16 update is small next to the base
weights that rounding realizes only about 64% of the trained delta (measured 2026-08-20, 38-79%
per module). A consumer that samples through vLLM needs a plain directory and pays that; a consumer
that only runs forward passes in-process does not have to. Applying the adapter at runtime computes
`Wx + B(Ax)` at full precision, and re-pointing one adapter name across a ladder of checkpoints
keeps the 4 GB base loaded once instead of once per checkpoint. `assert_one_adapter_config` is what
makes that safe, because PEFT reads an adapter's config only when the adapter *name* is new.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, NamedTuple, cast

import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors import safe_open
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from trl.trainer.utils import create_model_from_path

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

ADAPTER_CONFIG_FILENAME = "adapter_config.json"
CONFIG_FILENAME = "config.json"
PROCESSOR_CONFIG_FILENAME = "processor_config.json"

# The nested text config, present only on a composite checkpoint, which is what decides its layout.
COMPOSITE_CONFIG_KEY = "text_config"
CHECKPOINT_DIR_PATTERN = re.compile(r"^checkpoint-(\d+)$")

DEFAULT_MERGE_DTYPE = torch.bfloat16

# A segment of every adapter parameter's name, so the load and the key check must agree on it.
MERGE_ADAPTER_NAME = "default"

# Config fields that change what a checkpoint means; `assert_one_adapter_config` explains why.
ADAPTER_CONFIG_IDENTITY_FIELDS: tuple[str, ...] = (
    "base_model_name_or_path",
    "peft_type",
    "r",
    "lora_alpha",
    "target_modules",
    "modules_to_save",
    "use_dora",
    "use_rslora",
    "exclude_modules",
    "layers_to_transform",
    "rank_pattern",
    "alpha_pattern",
)

# Where a composite (vision-language) checkpoint nests its text tower, per transformers convention.
TEXT_SUBMODEL_KEY = "language_model"

# A merge writes here and moves into place only once read back and checked.
STAGING_SUFFIX = ".incomplete"

# How many offending names a mismatch error quotes; the counts beside them carry the magnitude.
ERROR_EXAMPLE_COUNT = 3


class MergedModel(NamedTuple):
    """A merged model with everything a consumer needs, and the count that proves it did work.

    `processor` is set only for a composite checkpoint, and it is not optional decoration there:
    a consumer picks its loader from the config, so a composite config sends vLLM looking for an
    image processor before it reads a single weight.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    applied_adapter_weights: int
    processor: ProcessorMixin | None


def checkpoint_step(checkpoint_dir: Path) -> int:
    """Return the training step a `checkpoint-<step>` directory holds."""
    match = CHECKPOINT_DIR_PATTERN.match(checkpoint_dir.name)
    if match is None:
        raise ValueError(
            f"{checkpoint_dir} is not a checkpoint directory; expected a name like 'checkpoint-40'."
        )
    return int(match.group(1))


def iter_checkpoints(run_dir: Path) -> Iterator[Path]:
    """Yield a run's checkpoint directories in training-step order.

    Sorted numerically, not lexically: with `save_steps=10` a run reaches `checkpoint-100`, which
    sorts before `checkpoint-20` as text and would quietly invert every before/after comparison
    built on this ordering. Non-checkpoint entries (`completions/`, `runs/`, loose files) are
    skipped rather than raising, since HF and TRL both write them next to the checkpoints.
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir {run_dir} is not a directory.")
    checkpoints = [
        path
        for path in run_dir.iterdir()
        if path.is_dir() and CHECKPOINT_DIR_PATTERN.match(path.name)
    ]
    yield from sorted(checkpoints, key=checkpoint_step)


def read_adapter_base_model(adapter_dir: Path) -> str:
    """Return the base model this adapter was trained against, as PEFT recorded it."""
    config_path = adapter_dir / ADAPTER_CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} not found, so {adapter_dir} is not a PEFT adapter directory."
        )
    config = json.loads(config_path.read_text())
    recorded = config.get("base_model_name_or_path")
    if not recorded:
        raise ValueError(
            f"{config_path} has no base_model_name_or_path, so the adapter cannot be matched to a "
            f"base model and merging it would be a guess."
        )
    return str(recorded)


def assert_adapter_matches_base(adapter_dir: Path, base_model_id: str) -> None:
    """Raise unless the adapter was trained against `base_model_id`.

    Compared as exact strings after trimming trailing slashes. A looser comparison would defeat
    the point: sibling models in one family differ only in a size suffix, share layer names and
    shapes, and so load onto each other's adapters without complaint.
    """
    recorded = read_adapter_base_model(adapter_dir).rstrip("/")
    if recorded != base_model_id.rstrip("/"):
        raise ValueError(
            f"Adapter {adapter_dir} was trained against {recorded!r}, not {base_model_id!r}. "
            f"Merging onto a different base of the same architecture succeeds silently and "
            f"produces a model that is not the one you trained."
        )


def assert_adapter_applies(adapted: PeftModel, adapter_dir: Path) -> int:
    """Raise unless every weight in the adapter file landed on a module of the loaded base.

    Returns the number of adapter weights that applied, which is the number the merge will fold in.

    PEFT matches adapter weights to modules by name. When the base is built with a different module
    tree than training used, nothing matches -- and PEFT's response is a `UserWarning`, because
    `PeftModel.from_pretrained` throws its load result away. `merge_and_unload` then merges nothing
    and the export is the untouched base model under a trained checkpoint's name, which no
    downstream consumer can detect. Reading that result is therefore the difference between a loud
    failure and a plausible null.

    `load_adapter` is the same load and it *returns* the result, so calling it is how the warning
    becomes an exception. It is idempotent for an adapter name already present -- it re-reads a
    tens-of-megabytes file and re-assigns the same tensors, which is nothing next to the base
    weights already in memory.
    """
    load_result = adapted.load_adapter(str(adapter_dir), MERGE_ADAPTER_NAME, is_trainable=False)
    unmatched = list(load_result.unexpected_keys)
    unfilled = list(load_result.missing_keys)
    injected = len(get_peft_model_state_dict(adapted, adapter_name=MERGE_ADAPTER_NAME))
    applied = injected - len(unfilled)
    if unmatched or unfilled:
        raise ValueError(
            f"Adapter {adapter_dir} did not apply to the base model built for it: {applied} of "
            f"{injected} adapter weights landed, {len(unmatched)} matched no module, and "
            f"{len(unfilled)} module slots were left unfilled. The base was built as "
            f"{type(adapted.get_base_model()).__name__}, whose module tree has to be the one "
            f"training used or the weight names cannot line up. Weights that matched nothing: "
            f"{unmatched[:ERROR_EXAMPLE_COUNT]}. Slots left unfilled: "
            f"{unfilled[:ERROR_EXAMPLE_COUNT]}. Merging in this state would write out the "
            f"unmodified base model under this checkpoint's name."
        )
    logger.info(f"adapter applied, {adapter_dir=} {applied=}")
    return applied


class AttachedAdapter(NamedTuple):
    """A model with one checkpoint's adapter live on it, and the count that proves it landed.

    `peft_model` is the wrapper, but the wrapped model is adapted *in place*: PEFT replaces the
    target `nn.Linear` modules inside the tree it was handed, so a forward pass driven through the
    original model object applies the adapter too. That is what lets an activation capture keep the
    base loaded once and re-point the adapter per checkpoint, and it is also the reason the count
    matters: nothing about the model object changes shape when an adapter fails to match.
    """

    peft_model: PeftModel
    applied_adapter_weights: int


def load_adapter_base(
    base_model_id: str,
    *,
    dtype: torch.dtype = DEFAULT_MERGE_DTYPE,
    device: torch.device | None = None,
) -> PreTrainedModel:
    """Build the base model the way training built it, ready for adapters to be attached.

    Same `create_model_from_path` as `load_merged_model` and for the same reason: the module tree
    the adapter names has to be the tree it gets, and `AutoModelForCausalLM` picks a different class
    for this family. `device_map` is passed explicitly as None because TRL fills it with "auto" when
    the key is absent, which shards the model across every visible card; the move to one device is
    done here instead, so a single-GPU capture stays on a single GPU.
    """
    built: PreTrainedModel = create_model_from_path(
        base_model_id, dtype=dtype, device_map=None, trust_remote_code=True
    )
    # transformers 5.x wraps `.to`, which pyright reads as an unbound __call__ wanting `self`.
    placed = built if device is None else built.to(device)  # pyright: ignore[reportArgumentType]
    return placed.eval()


def attach_adapter(
    model: PreTrainedModel,
    adapter_dir: Path,
    base_model_id: str,
    *,
    existing: PeftModel | None = None,
) -> AttachedAdapter:
    """Attach one checkpoint's adapter to `model` at runtime, and prove it landed.

    Pass `existing` to re-point an already-wrapped model at another checkpoint: the adapter is
    re-loaded under the same name, which overwrites its tensors in place rather than rebuilding the
    base. `assert_adapter_applies` runs on both paths, since a name mismatch is the failure that
    otherwise captures pristine base activations under a trained checkpoint's name.

    Re-pointing is only sound while every checkpoint shares one adapter config -- see
    `assert_one_adapter_config`, which the caller should run over its whole ladder up front.
    """
    assert_adapter_matches_base(adapter_dir, base_model_id)
    adapted = (
        existing
        if existing is not None
        else PeftModel.from_pretrained(model, str(adapter_dir), adapter_name=MERGE_ADAPTER_NAME)
    )
    applied = assert_adapter_applies(adapted, adapter_dir)
    adapted.set_adapter(MERGE_ADAPTER_NAME)
    return AttachedAdapter(peft_model=adapted, applied_adapter_weights=applied)


def canonical_config_value(value: object) -> object:
    """Return one config value in a form that compares by content rather than by write order.

    PEFT holds `target_modules` and its list-valued siblings as sets and serialises them as JSON
    arrays, so one unchanged adapter config saved at successive checkpoints can differ in array
    order alone (three orderings observed across the 14 checkpoints of one ladder). Comparing the
    raw arrays reads that as a changed target list and refuses a ladder that is in fact uniform.
    A regex `target_modules` arrives as a str and keeps its exact value, since order carries
    meaning there.
    """
    if isinstance(value, list):
        return tuple(sorted(cast("list[str]", value)))
    return value


def adapter_config_identity(adapter_dir: Path) -> dict[str, object]:
    """Return the adapter-config fields that decide what a checkpoint means, order-normalised."""
    raw = (adapter_dir / ADAPTER_CONFIG_FILENAME).read_text()
    config = cast("dict[str, object]", json.loads(raw))
    return {
        field: canonical_config_value(config.get(field)) for field in ADAPTER_CONFIG_IDENTITY_FIELDS
    }


def assert_one_adapter_config(adapter_dirs: Sequence[Path]) -> None:
    """Raise unless every adapter in a ladder shares one config.

    A capture or eval that re-points a single PEFT adapter name across checkpoints depends on this:
    PEFT reads an adapter's config only when the name is new, so a later checkpoint with a different
    rank or target list is loaded under the *first* one's config. A rank change then matches shapes
    only by accident and a target-list change leaves modules unadapted, both without a warning.
    """
    reference: dict[str, object] | None = None
    reference_dir: Path | None = None
    for adapter_dir in adapter_dirs:
        identity = adapter_config_identity(adapter_dir)
        if reference is None:
            reference, reference_dir = identity, adapter_dir
            continue
        differing = sorted(field for field in identity if identity[field] != reference[field])
        if differing:
            raise ValueError(
                f"Adapter configs differ between {reference_dir} and {adapter_dir} on {differing}. "
                f"A ladder that re-points one PEFT adapter name keeps the first config, so the "
                f"later checkpoints would be applied under the wrong one with nothing going red."
            )


def is_composite(config: object) -> bool:
    """Whether a config describes a composite (vision-language) checkpoint rather than a text one.

    Keyed on the presence of a nested text config, which is the thing that actually decides the
    layout: the composite class builds its text tower from `config.text_config`, and so nests every
    text weight beneath it. Reading the architecture name instead would need a list of class names
    per model family, which rots the moment a sibling ships.
    """
    if isinstance(config, dict):
        return COMPOSITE_CONFIG_KEY in config
    return hasattr(config, COMPOSITE_CONFIG_KEY)


def read_export_weight_names(model_dir: Path) -> list[str]:
    """Return the weight names a consumer reads out of a saved model directory.

    Read off the files rather than the model in memory, because the two disagree: transformers
    renames keys as it writes, so a flat runtime tree can land on disk under nested names. Only the
    written names decide whether a consumer can load the directory.
    """
    index_path = model_dir / SAFE_WEIGHTS_INDEX_NAME
    if index_path.is_file():
        index = cast("dict[str, dict[str, str]]", json.loads(index_path.read_text()))
        return sorted(index["weight_map"])
    shard_path = model_dir / SAFE_WEIGHTS_NAME
    if not shard_path.is_file():
        raise FileNotFoundError(
            f"{model_dir} holds neither {SAFE_WEIGHTS_INDEX_NAME} nor {SAFE_WEIGHTS_NAME}, so it "
            f"is not a saved model directory and its weight names cannot be read."
        )
    with safe_open(str(shard_path), framework="pt") as shard:
        return sorted(cast("Iterator[str]", shard.keys()))


def assert_export_is_self_consistent(model_dir: Path) -> None:
    """Raise unless the config's architecture and the weight names on disk describe one model.

    Two layouts are coherent. A composite checkpoint's config carries a `text_config` and its
    weights nest the text tower under `language_model`; a text-only checkpoint has no `text_config`
    and its weights are flat. A directory mixing the two loads under neither reading, because a
    consumer picks its loader from `architectures` and then cannot find the nesting the weights
    actually use -- vLLM says only that some module does not exist. That mixture is what a merge
    produces when the class it loaded is not the class whose layout it writes.

    A composite config also has to be paid for in full. It tells a consumer this is a
    vision-language checkpoint, and vLLM then builds a multimodal processor *before* it reads any
    weight, so an export carrying only a model and a tokenizer fails on a missing image processor --
    an error that names the processor and says nothing about the merge that omitted it.
    """
    config_path = model_dir / CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} not found, so {model_dir} has no architecture.")
    config = cast("dict[str, object]", json.loads(config_path.read_text()))
    weight_names = read_export_weight_names(model_dir)
    composite_config = is_composite(config)
    nested_weights = any(f".{TEXT_SUBMODEL_KEY}." in name for name in weight_names)
    if composite_config != nested_weights:
        raise ValueError(
            f"{model_dir} is not internally consistent: its config names "
            f"{config.get('architectures')} and {'carries' if composite_config else 'has no'} "
            f"a text_config, while its weight names are "
            f"{'nested' if nested_weights else 'flat'} (for example {weight_names[0]!r}). A "
            f"composite config needs weights nested under {TEXT_SUBMODEL_KEY!r} and a text-only "
            f"config needs flat ones; nothing can load a directory that mixes them."
        )
    if composite_config and not (model_dir / PROCESSOR_CONFIG_FILENAME).is_file():
        raise ValueError(
            f"{model_dir} declares the composite architecture {config.get('architectures')} but "
            f"holds no {PROCESSOR_CONFIG_FILENAME}. A consumer reads that config and builds a "
            f"multimodal processor before it touches a weight, so it will fail looking for an image "
            f"processor rather than saying anything about this directory being incomplete."
        )


def load_merged_model(
    adapter_dir: Path,
    base_model_id: str,
    *,
    dtype: torch.dtype = DEFAULT_MERGE_DTYPE,
    device_map: str | None = None,
) -> MergedModel:
    """Load the base model the way training did, apply the adapter, and merge it in.

    The base goes through TRL's `create_model_from_path` -- the same call `GRPOTrainer` makes -- so
    the module tree the adapter names is the tree it gets, by construction. Building it any other
    way is not a style choice: `AutoModelForCausalLM` picks a different class for this family and
    every adapter name then misses.

    `dtype` rather than the deprecated `torch_dtype`: transformers 5.x ignores the old name
    silently, which would load the whole thing in float32 and blow the memory budget without
    saying so. `device_map` is passed explicitly even when None, because TRL fills it with "auto"
    when the key is absent and that shards the model across every visible card. The tokenizer comes
    back alongside the model because a merged checkpoint is only consumable if both are written.
    """
    assert_adapter_matches_base(adapter_dir, base_model_id)
    logger.info(f"merging adapter, {adapter_dir=} {base_model_id=} {dtype=} {device_map=}")
    base = create_model_from_path(
        base_model_id, dtype=dtype, device_map=device_map, trust_remote_code=True
    )
    adapted = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name=MERGE_ADAPTER_NAME)
    applied = assert_adapter_applies(adapted, adapter_dir)
    # merge_and_unload lives on the tuner mixin reached through PeftModel.__getattr__.
    merged = cast(
        "PreTrainedModel",
        adapted.merge_and_unload(),  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
    )
    tokenizer = cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True),
    )
    processor = (
        cast(
            "ProcessorMixin",
            AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True),
        )
        if is_composite(merged.config)
        else None
    )
    return MergedModel(
        model=merged,
        tokenizer=tokenizer,
        applied_adapter_weights=applied,
        processor=processor,
    )


def export_merged_checkpoint(
    adapter_dir: Path,
    base_model_id: str,
    out_dir: Path,
    *,
    dtype: torch.dtype = DEFAULT_MERGE_DTYPE,
    device_map: str | None = None,
) -> Path:
    """Write a merged adapter out as a plain HF model directory and return that path.

    The result is an ordinary checkpoint: `HFBackend`, `VLLMBackend`, and the interp stack consume
    it with no adapter awareness at all.

    Written to a staging directory and moved into place only after being read back and checked, so
    nothing at the consumer's path is ever half-written or unvalidated. A staging directory left
    behind is the evidence from a failed merge, so it is not cleaned up and not written over.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already exists and is not empty; refusing to overwrite a checkpoint. "
            f"Merged exports are cheap to regenerate, so a stale one is not worth the ambiguity."
        )
    staging_dir = out_dir.with_name(out_dir.name + STAGING_SUFFIX)
    if staging_dir.exists():
        raise FileExistsError(
            f"{staging_dir} is left over from an export that did not finish. Read it before "
            f"removing it: it is what the failed merge wrote, and so it is the evidence for why."
        )
    merged = load_merged_model(adapter_dir, base_model_id, dtype=dtype, device_map=device_map)
    staging_dir.mkdir(parents=True)
    merged.model.save_pretrained(str(staging_dir))
    merged.tokenizer.save_pretrained(str(staging_dir))
    if merged.processor is not None:
        merged.processor.save_pretrained(str(staging_dir))
    assert_export_is_self_consistent(staging_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in staging_dir.iterdir():
        path.rename(out_dir / path.name)
    staging_dir.rmdir()
    logger.info(
        f"merged checkpoint written, {out_dir=} {adapter_dir=} "
        f"applied={merged.applied_adapter_weights}"
    )
    return out_dir
